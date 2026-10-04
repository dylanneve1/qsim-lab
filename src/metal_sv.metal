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
// TG_BITS (max inner qubits, threadgroup buffer size) and TPG (threads per
// threadgroup) are set by the host when it compiles this source.

#include <metal_stdlib>
using namespace metal;

#ifndef TG_BITS
#define TG_BITS 12
#endif
#ifndef TPG
#define TPG 512
#endif
#ifndef TPG_BITS
#define TPG_BITS 9
#endif
#define TG_SIZE (1u << TG_BITS)
// elements, pairs and quads per thread (TPG <= TG_SIZE / 2)
#define EPT (TG_SIZE / TPG)
#define PPT (EPT / 2)
#define QPT ((EPT / 4) > 0 ? (EPT / 4) : 1)

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
//       5 batch of t uncontrolled 2x2 gates (targets packed in a, matrices
//       at mats[2 * nterms ...])
struct OpG {
    uint kind;
    uint t;      // target buffer bit (kinds 0-2); swap: lower bit
    uint cin;    // control mask on buffer bits
    uint omask;  // outer control mask (physical bits)
    uint opat;   // outer control pattern
    uint a;      // diag: first group; swap: upper bit
    uint nterms; // diag: number of groups
    uint pad;
    float m[8];  // row-major m00 m01 m10 m11 as (re, im) pairs
};

// A pivot group of a diagonal run: the factor
// c * tabs[tab + (e & 63)] * tabs[tab + 64 + (e >> 6)] (complex) applies to
// buffer indices with (e & pmask) == ppat, where c is the product of the
// group's outer factors whose outer condition holds for this threadgroup.
struct Group {
    uint pmask;
    uint ppat;
    uint tab;    // offset into tabs (128 entries), if lin
    uint o0;     // outer factors oterms[o0 .. o0 + on]
    uint on;
    uint lin;    // tables present
    uint pad0;
    uint pad1;
};

// Outer factor of a group: f when (base & omask) == opat.
struct OTerm {
    uint omask;
    uint opat;
    float2 f;
};

static inline uint insert0(uint p, uint t) {
    return ((p >> t) << (t + 1)) | (p & ((1u << t) - 1u));
}

static inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// K uncontrolled 2x2 gates on the distinct buffer bits packed (8 bits each,
// ascending) in `packed`; matrix j is mats[2j] = (m00, m01), mats[2j+1] =
// (m10, m11). Each thread keeps the 2^K amplitudes of a group in registers.
template <uint K>
static inline void batch(threadgroup float2* buf, uint size, uint tid, uint packed,
                         device const float4* mats)
{
    constexpr uint M = 1u << K;
    constexpr uint GPT = (EPT / M) > 0 ? (EPT / M) : 1;
    uint tq[K];
    for (uint j = 0; j < K; j++) tq[j] = (packed >> (8 * j)) & 0xffu;
    const uint ngroups = size >> K;
    for (uint gi = 0; gi < GPT; gi++) {
        const uint gidx = tid + gi * TPG;
        if (gidx >= ngroups) break;
        uint b0 = gidx;
        for (uint j = 0; j < K; j++) b0 = insert0(b0, tq[j]);
        uint off[M];
        float2 v[M];
        for (uint m = 0; m < M; m++) {
            uint idx = b0;
            for (uint j = 0; j < K; j++) {
                if ((m >> j) & 1u) idx |= 1u << tq[j];
            }
            off[m] = idx;
            v[m] = buf[idx];
        }
        for (uint j = 0; j < K; j++) {
            const float4 r0 = mats[2 * j];
            const float4 r1 = mats[2 * j + 1];
            for (uint m = 0; m < M; m++) {
                if ((m >> j) & 1u) continue;
                const uint m1 = m | (1u << j);
                const float2 x = v[m];
                const float2 y = v[m1];
                v[m] = cmul(r0.xy, x) + cmul(r0.zw, y);
                v[m1] = cmul(r1.xy, x) + cmul(r1.zw, y);
            }
        }
        for (uint m = 0; m < M; m++) buf[off[m]] = v[m];
    }
}

kernel void stage(device float2* psi [[buffer(0)]],
                  constant StageHdr& h [[buffer(1)]],
                  device const OpG* ops [[buffer(2)]],
                  device const Group* groups [[buffer(3)]],
                  device const float2* tabs [[buffer(4)]],
                  device const OTerm* oterms [[buffer(5)]],
                  device const float4* mats [[buffer(6)]],
                  uint g [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float2 buf[TG_SIZE];
    uint base = 0;
    for (uint k = 0; k < h.nout; k++) {
        base |= ((g >> k) & 1u) << h.outer[k];
    }
    const uint size = 1u << h.l;
    const uint b = h.b;
    const uint lowmask = (1u << b) - 1u;
    for (uint i = 0; i < EPT; i++) {
        const uint e = tid + i * TPG;
        if (e < size) buf[e] = psi[base + (e & lowmask) + h.hioff[e >> b]];
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
            for (uint i = 0; i < PPT; i++) {
                const uint p = tid + i * TPG;
                if (p >= half_) break;
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
            for (uint i = 0; i < PPT; i++) {
                const uint p = tid + i * TPG;
                if (p >= half_) break;
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
            for (uint i = 0; i < PPT; i++) {
                const uint p = tid + i * TPG;
                if (p >= half_) break;
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
            for (uint i = 0; i < QPT; i++) {
                const uint q = tid + i * TPG;
                if (q >= (size >> 2)) break;
                const uint j = insert0(insert0(q, a), bb);
                const uint x = j | (1u << a);
                const uint y = j | (1u << bb);
                const float2 v = buf[x];
                buf[x] = buf[y];
                buf[y] = v;
            }
            break;
        }
        case 4: {
            // diagonal run: op.nterms pivot groups starting at op.a.
            float2 acc[EPT];
            for (uint i = 0; i < EPT; i++) acc[i] = float2(1.0f, 0.0f);
            const uint g0 = op.a, g1 = op.a + op.nterms;
            for (uint k = g0; k < g1; k++) {
                const Group gr = groups[k];
                float2 c = float2(1.0f, 0.0f);
                for (uint j = gr.o0; j < gr.o0 + gr.on; j++) {
                    const OTerm ot = oterms[j];
                    if ((base & ot.omask) == ot.opat) c = cmul(c, ot.f);
                }
                if (gr.lin == 0u && c.x == 1.0f && c.y == 0.0f) continue;
                for (uint i = 0; i < EPT; i++) {
                    const uint e = tid + i * TPG;
                    if ((e & gr.pmask) == gr.ppat) {
                        float2 f = c;
                        if (gr.lin != 0u) {
                            f = cmul(f, cmul(tabs[gr.tab + (e & 63u)],
                                             tabs[gr.tab + 64u + (e >> 6)]));
                        }
                        acc[i] = cmul(acc[i], f);
                    }
                }
            }
            for (uint i = 0; i < EPT; i++) {
                const uint e = tid + i * TPG;
                if (e < size) buf[e] = cmul(buf[e], acc[i]);
            }
            break;
        }
        case 5: {
            // batch of op.t uncontrolled 2x2 gates on distinct buffer bits
            switch (op.t) {
            case 2: batch<2>(buf, size, tid, op.a, mats + 2 * op.nterms); break;
            case 3: batch<3>(buf, size, tid, op.a, mats + 2 * op.nterms); break;
            case 4: batch<4>(buf, size, tid, op.a, mats + 2 * op.nterms); break;
            default: break;
            }
            break;
        }
        default:
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < EPT; i++) {
        const uint e = tid + i * TPG;
        if (e < size) psi[base + (e & lowmask) + h.hioff[e >> b]] = buf[e];
    }
}

// ----- register-resident stage kernel ------------------------------------
//
// Same stage semantics as `stage`, but thread `tid` keeps buffer elements
// e = tid + i * TPG (i < EPT) in registers for the whole stage. A gate on
// buffer bit t is then
//   t >= TPG_BITS            : register-local (no barrier, no shared memory)
//   t <  LANE_BITS (<= 5)    : simd_shuffle_xor within the SIMD group
//   LANE_BITS <= t < TPG_BITS: exchange through threadgroup memory
// Requires 2^h.l == TG_SIZE (the host compiles one pipeline per l).

#define EPT_BITS (TG_BITS - TPG_BITS)
#define LANE_BITS (TPG_BITS < 5 ? TPG_BITS : 5)
#define LANES (1u << LANE_BITS)

template <uint B>
static inline void loc_u1(thread float2* v, uint tid, uint cin,
                          float2 m00, float2 m01, float2 m10, float2 m11)
{
    for (uint i = 0; i < EPT; i++) {
        if (i & (1u << B)) continue;
        const uint i1 = i | (1u << B);
        const uint e0 = tid + i * TPG;
        if ((e0 & cin) != cin) continue;
        const float2 x = v[i];
        const float2 y = v[i1];
        v[i] = cmul(m00, x) + cmul(m01, y);
        v[i1] = cmul(m10, x) + cmul(m11, y);
    }
}

kernel void stage_reg(device float2* psi [[buffer(0)]],
                      constant StageHdr& h [[buffer(1)]],
                      device const OpG* ops [[buffer(2)]],
                      device const Group* groups [[buffer(3)]],
                      device const float2* tabs [[buffer(4)]],
                      device const OTerm* oterms [[buffer(5)]],
                      uint g [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float2 xbuf[TG_SIZE];
    uint base = 0;
    for (uint k = 0; k < h.nout; k++) {
        base |= ((g >> k) & 1u) << h.outer[k];
    }
    const uint b = h.b;
    const uint lowmask = (1u << b) - 1u;
    float2 v[EPT];
    for (uint i = 0; i < EPT; i++) {
        const uint e = tid + i * TPG;
        v[i] = psi[base + (e & lowmask) + h.hioff[e >> b]];
    }

    for (uint o = 0; o < h.nops; o++) {
        const OpG op = ops[o];
        if ((base & op.omask) != op.opat) {
            continue;
        }
        const uint t = op.t;
        const uint cin = op.cin;
        switch (op.kind) {
        case 0:
        case 1:
        case 2: {
#ifdef DBG_NO_U1
            break;
#endif
            const float2 m00 = float2(op.m[0], op.m[1]);
            const float2 m01 = float2(op.m[2], op.m[3]);
            const float2 m10 = float2(op.m[4], op.m[5]);
            const float2 m11 = float2(op.m[6], op.m[7]);
            if (t >= TPG_BITS) {
                switch (t - TPG_BITS) {
#if EPT_BITS > 0
                case 0: loc_u1<0>(v, tid, cin, m00, m01, m10, m11); break;
#endif
#if EPT_BITS > 1
                case 1: loc_u1<1>(v, tid, cin, m00, m01, m10, m11); break;
#endif
#if EPT_BITS > 2
                case 2: loc_u1<2>(v, tid, cin, m00, m01, m10, m11); break;
#endif
#if EPT_BITS > 3
                case 3: loc_u1<3>(v, tid, cin, m00, m01, m10, m11); break;
#endif
#if EPT_BITS > 4
                case 4: loc_u1<4>(v, tid, cin, m00, m01, m10, m11); break;
#endif
#if EPT_BITS > 5
                case 5: loc_u1<5>(v, tid, cin, m00, m01, m10, m11); break;
#endif
#if EPT_BITS > 6
                case 6: loc_u1<6>(v, tid, cin, m00, m01, m10, m11); break;
#endif
                default: break;
                }
            } else {
                const bool hi = ((tid >> t) & 1u) != 0u;
                const float2 a0 = hi ? m11 : m00;  // factor on own value
                const float2 a1 = hi ? m10 : m01;  // factor on partner
                if (t < LANE_BITS) {
                    const ushort sh = ushort(1u << t);
                    for (uint i = 0; i < EPT; i++) {
                        const float2 pv = simd_shuffle_xor(v[i], sh);
                        const uint e = tid + i * TPG;
                        if ((e & cin) == cin) v[i] = cmul(a0, v[i]) + cmul(a1, pv);
                    }
                } else {
                    const uint sh = 1u << t;
                    for (uint i = 0; i < EPT; i++) xbuf[tid + i * TPG] = v[i];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint i = 0; i < EPT; i++) {
                        const uint e = tid + i * TPG;
                        const float2 pv = xbuf[e ^ sh];
                        if ((e & cin) == cin) v[i] = cmul(a0, v[i]) + cmul(a1, pv);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            break;
        }
        case 3: {
            const uint a = t, bb = op.a;
            for (uint i = 0; i < EPT; i++) xbuf[tid + i * TPG] = v[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = 0; i < EPT; i++) {
                const uint e = tid + i * TPG;
                const uint d = ((e >> a) ^ (e >> bb)) & 1u;
                v[i] = xbuf[d ? (e ^ ((1u << a) | (1u << bb))) : e];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            break;
        }
        case 4: {
#ifdef DBG_NO_DIAG
            break;
#endif
            // factors multiply straight into v (no accumulator array: keeps
            // register pressure at EPT float2)
            const uint g0 = op.a, g1 = op.a + op.nterms;
            for (uint k = g0; k < g1; k++) {
                const Group gr = groups[k];
                // product of the outer factors, LANES at a time: each lane
                // evaluates one term, then a butterfly over the SIMD group
                float2 c = float2(1.0f, 0.0f);
#ifndef DBG_NO_OTERMS
                for (uint j0 = gr.o0; j0 < gr.o0 + gr.on; j0 += LANES) {
                    const uint j = j0 + (tid & (LANES - 1u));
                    float2 f = float2(1.0f, 0.0f);
                    if (j < gr.o0 + gr.on) {
                        const OTerm ot = oterms[j];
                        if ((base & ot.omask) == ot.opat) f = ot.f;
                    }
                    for (ushort sh = 1; sh < LANES; sh <<= 1) {
                        f = cmul(f, simd_shuffle_xor(f, sh));
                    }
                    c = cmul(c, f);
                }
#endif
#ifdef DBG_NO_LIN
                continue;
#endif
                if (gr.lin == 0u) {
                    if (c.x == 1.0f && c.y == 0.0f) continue;
                    for (uint i = 0; i < EPT; i++) {
                        const uint e = tid + i * TPG;
                        const bool on = (e & gr.pmask) == gr.ppat;
                        v[i] = on ? cmul(v[i], c) : v[i];
                    }
                    continue;
                }
                device const float2* t0 = tabs + gr.tab;
                device const float2* t1 = tabs + gr.tab + 64u;
#if TPG >= 64
                // e & 63 == tid & 63 for every element of this thread
                const float2 c0 = cmul(c, t0[tid & 63u]);
#endif
                for (uint i = 0; i < EPT; i++) {
                    const uint e = tid + i * TPG;
#if TPG >= 64
                    const float2 f = cmul(c0, t1[e >> 6]);
#else
                    const float2 f = cmul(c, cmul(t0[e & 63u], t1[e >> 6]));
#endif
                    const bool on = (e & gr.pmask) == gr.ppat;
                    v[i] = on ? cmul(v[i], f) : v[i];
                }
            }
            break;
        }
        default:
            break;
        }
    }

    for (uint i = 0; i < EPT; i++) {
        const uint e = tid + i * TPG;
        psi[base + (e & lowmask) + h.hioff[e >> b]] = v[i];
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
