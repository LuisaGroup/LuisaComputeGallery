// CUDA C++ source template of the Qwen3.5-4B causal GQA PREFILL ATTENTION
// (opt_plan/09 N1b, FlashAttention-2 style). Compiled at run time by
// `NativeShaderExt` (`NativeShaderLanguage::CUDA_NVRTC`) on the CUDA backend; no
// other backend has this route, and nothing here is part of the DSL/AST pipeline.
//
// Contract of the kernel (identical to `make_qwen35_attn_prefill_tiled_kernel` and
// to the reference `make_qwen_attn_prefill_kernel`):
//   q        : [tokens, q_heads, head_dim]   FP32 queries
//   gate     : [tokens, q_heads, head_dim]   FP32 sigmoid output gate
//   k_cache  : [capacity, kv_heads, head_dim] FP32 keys of the paged KV cache
//   v_cache  : [capacity, kv_heads, head_dim] FP32 values
//   out      : [tokens, q_heads, head_dim]   FP32 out, row-major
//   tokens, scale, pos0, kv_len, capacity : runtime geometry
// Query row t sits at absolute position pos0 + t and attends keys
// [0, min(pos0 + t, kv_len - 1)], i.e. the same causal window the DSL kernels use.
// Rows >= tokens, keys >= kv_len and kv_len > capacity are defensive bounds, not
// program invariants: the kernel zero-fills the smem tile and masks per score, so
// a ragged tail or a clamped length can never read stale shared memory.
//
// ALGORITHM (FlashAttention-2, one pass, no S x S materialization)
//   One 256-thread block per (16-token query tile, kv head). The 8 warps are the
//   2 x 4 GQA pack: warp w owns q head `kv_head*4 + (w >> 1)` and head-dim half
//   `(w & 1) * 128`, so every K/V tile staged by the block serves all 4 q heads
//   (the 4:1 GQA dedup) and the FP32 output accumulator of a warp is
//   [16 rows x 128 channels] = 64 registers per lane, which is what makes
//   head_dim = 256 fit a hand-written mma kernel at all.
//   Per KV tile (32 keys): smem K/V tiles are loaded as FP32 from the cache,
//   rounded to BF16 through registers (`cvt.rn.bf16.f32`) and stored to shared
//   memory as BF16 rows; the QK^T partial each warp can see (its 128-deep half)
//   goes through `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, is
//   exchanged with its sister warp through ONE shared [16 x 32] score buffer,
//   summed, scaled and causally masked, then the FP32 online softmax
//   (running max / running sum / one accumulator rescale per tile) produces the
//   BF16 P fragments in the mma A layout and P.V runs in the same accumulators.
//   The staging of the NEXT tile is issued into registers before that P.V so its
//   global latency is covered by mma work.
//
// NUMERICS (documented, not hidden)
//   * Q and K are rounded FP32 -> BF16 once per block (each element is rounded
//     identically for every key tile, so the score is a deterministic function of
//     the BF16 operands), and the QK^T as well as P.V accumulations are FP32.
//   * P = exp2((S - m) * log2 e) is computed in FP32 and rounded to BF16 for the
//     P.V mma (the shared-memory P is the mma operand, exactly like FlashAttention
//     stores P in half precision); the online-softmax denominator and the running
//     max stay FP32.
//   * exp() is `ex2.approx.f32` (<= 2^-22 relative), the same exp2 substitution
//     the DSL tiled kernel makes.
//   * Therefore this path is NOT bit-comparable with the FP32 DSL tiled kernel:
//     see docs/examples/example_qwen.md for the measured deviation (relative RMS
//     vs the FP32 reference and vs ref_qwen_attn_prefill) and the tolerance the
//     differential test enforces.
#ifndef QWEN_NATIVE_ATTN_KERNEL_H
#define QWEN_NATIVE_ATTN_KERNEL_H

namespace luisa::qwen::dsl {

// `@NAME@` tokens are replaced by qwen_native_attn_cuda_source() (host) and by the
// standalone nvcc harness. Keep this a raw literal: the engine hands it to NVRTC
// verbatim after substitution.
inline constexpr char qwen_native_attn_cuda_template[] = R"QNATTN(
// Qwen3.5-4B causal GQA prefill attention, BF16 tensor cores (FlashAttention-2).
// GENERATED from src/qwen/qwen/native_attn_kernel.h - do not edit.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#error "qwen native prefill attention needs sm_80 or newer: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
#endif

#define QN_HD @HD@
#define QN_QH @QHEADS@
#define QN_KVH @KVHEADS@
#define QN_GQA @GQA@
#define QN_QT @QTILE@
#define QN_KT @KVTILE@
#define QN_WARPS @WARPS@
#define QN_BLOCK @BLOCK@
// Head-dim columns one warp owns (the other half belongs to its sister warp).
#define QN_HALF (QN_HD / 2)
// Shared K/V row stride in bf16 elements: one 256-element row plus 8 elements
// (16 B) of padding. The padding keeps every row start 16-byte aligned for
// ldmatrix AND spreads eight consecutive rows over eight distinct bank groups.
#define QN_HSTRIDE (QN_HD + 8)
// Shared score row stride in floats (one 32-key row + 1 float of padding so
// consecutive rows land on different banks).
#define QN_SSTRIDE (QN_KT + 1)
// k steps of the QK^T gemm inside one warp's 128-deep half.
#define QN_KSTEPS (QN_HALF / 16)
// n8 fragments (keys) of the QK^T gemm and (channels) of the P.V gemm.
#define QN_NQK (QN_KT / 8)
#define QN_KPV (QN_KT / 16)
#define QN_NPV (QN_HALF / 8)
// Global row strides: [tokens, q_heads, hd] for q/gate/out, [capacity, kv_heads,
// hd] for the cache. Both are compile-time here, like the DSL tiled kernel.
#define QN_QSTRIDE (QN_QH * QN_HD)
#define QN_KVGSTRIDE (QN_KVH * QN_HD)
// FP32 score sentinel of a masked (row, key) pair. Finite on purpose: the online
// softmax must not see inf * 0 or inf - inf, and two halves of a masked score sum
// to -2e30, which still underflows exp2 to zero.
#define QN_FLOOR (-1.0e30f)
#define QN_LOG2E 1.4426950408889634f
// Threshold past which sigmoid(x) is evaluated as 1 / (1 + exp(-x)) in FP32.
#define QN_NEG_INF (-3.0e38f)

static_assert(QN_HD == 256, "this realization is the head_dim 256 model geometry");
static_assert(QN_GQA == 4 && QN_WARPS == 2 * QN_GQA, "one 2-warp pair per q head of a kv head");
static_assert(QN_HD % 32 == 0 && QN_HALF % 16 == 0, "the head-dim half must tile the m16n8k16 k steps");
static_assert(QN_KT % 16 == 0 && QN_KT % 8 == 0, "the key tile tiles both gemms");
static_assert(QN_BLOCK == QN_WARPS * 32, "one 32-lane warp per (q head, head-dim half)");
static_assert(QN_HSTRIDE % 8 == 0, "ldmatrix rows must stay 16-byte aligned");
static_assert(QN_QT == 16, "the query tile is one m16 fragment per warp");

// ---------------------------------------------------------------------------
// PTX helpers (no CUDA headers: NVRTC compiles this translation unit standalone)
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned qn_smem_u32(const void *ptr) {
    unsigned addr;
    asm("{\n\t"
        ".reg .u64 qn_smem_ptr;\n\t"
        "cvta.to.shared.u64 qn_smem_ptr, %1;\n\t"
        "cvt.u32.u64 %0, qn_smem_ptr;\n\t"
        "}"
        : "=r"(addr)
        : "l"(ptr));
    return addr;
}

__device__ __forceinline__ void qn_ldg4(float &f0, float &f1, float &f2, float &f3,
                                        const float *p) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                 : "=f"(f0), "=f"(f1), "=f"(f2), "=f"(f3)
                 : "l"(p));
}

__device__ __forceinline__ void qn_ldg2(float &f0, float &f1, const float *p) {
    asm volatile("ld.global.v2.f32 {%0,%1}, [%2];\n" : "=f"(f0), "=f"(f1) : "l"(p));
}

__device__ __forceinline__ void qn_stg2(float *p, float f0, float f1) {
    asm volatile("st.global.v2.f32 [%0], {%1,%2};\n" ::"l"(p), "f"(f0), "f"(f1));
}

__device__ __forceinline__ unsigned qn_bf16x2(float lo, float hi) {
    unsigned short a, b;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(a) : "f"(lo));
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(b) : "f"(hi));
    return (unsigned)a | ((unsigned)b << 16);
}

// exp2 with the hardware EX2 unit (<= 2^-22 relative error).
__device__ __forceinline__ float qn_ex2(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// ---- 1 / (1 + exp(-x)) with the same exp2 substitution ---------------------
__device__ __forceinline__ float qn_sigmoid(float x) {
    return 1.0f / (1.0f + qn_ex2(-x * QN_LOG2E));
}

__device__ __forceinline__ float qn_max2(float a, float b) { return a > b ? a : b; }

// Butterfly reduction inside a 4-lane quad (the 4 lanes that share one m16 row).
__device__ __forceinline__ float qn_quad_max(float v) {
    float t;
    asm volatile("shfl.sync.bfly.b32 %0, %1, 1, 0x1f, 0xffffffff;" : "=f"(t) : "f"(v));
    v = qn_max2(v, t);
    asm volatile("shfl.sync.bfly.b32 %0, %1, 2, 0x1f, 0xffffffff;" : "=f"(t) : "f"(v));
    return qn_max2(v, t);
}

__device__ __forceinline__ float qn_quad_sum(float v) {
    float t;
    asm volatile("shfl.sync.bfly.b32 %0, %1, 1, 0x1f, 0xffffffff;" : "=f"(t) : "f"(v));
    v = v + t;
    asm volatile("shfl.sync.bfly.b32 %0, %1, 2, 0x1f, 0xffffffff;" : "=f"(t) : "f"(v));
    return v + t;
}

__device__ __forceinline__ void qn_ldmatrix_x2(unsigned (&r)[2], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(addr));
}

__device__ __forceinline__ void qn_ldmatrix_x2_trans(unsigned (&r)[2], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(addr));
}

// D += A * B with A[16,16] row-major, B[16,8] col-major, FP32 accumulate.
__device__ __forceinline__ void qn_mma(float (&d)[4], const unsigned (&a)[4],
                                       const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------------------
// staging
// ---------------------------------------------------------------------------
// K/V of one kv head, keys [k0, k0+QN_KT), are contiguous 256-float rows. They are
// read as FP32 float4 (16 B, fully coalesced), rounded to BF16 in registers and
// written to the shared bf16 tile as two 32-bit words (8 B) per chunk. Out of
// range keys (the ragged tail of the causal window, or a defensive clamp) are
// zero-filled: the score mask drops them anyway, so no lane can ever observe stale
// shared memory.
__device__ __forceinline__ void qn_load_kv(unsigned (&kreg)[2 * (QN_KT * QN_HD) / (4 * QN_BLOCK)],
                                           unsigned (&vreg)[2 * (QN_KT * QN_HD) / (4 * QN_BLOCK)],
                                           const float *__restrict__ k_cache,
                                           const float *__restrict__ v_cache,
                                           unsigned tid, unsigned kv_head, unsigned k0,
                                           unsigned key_end) {
    constexpr unsigned chunks = QN_HD / 4;// float4 chunks of one key row
    constexpr unsigned steps = (QN_KT * QN_HD) / (4u * QN_BLOCK);
#pragma unroll
    for (unsigned s = 0u; s < steps; ++s) {
        const unsigned c = tid + s * QN_BLOCK;
        const unsigned r = c / chunks;
        const unsigned hd = (c % chunks) * 4u;
        const unsigned key = k0 + r;
        float f[4];
        if (key < key_end) {
            const size_t base = (size_t)key * QN_KVGSTRIDE + kv_head * QN_HD + hd;
            qn_ldg4(f[0], f[1], f[2], f[3], k_cache + base);
            kreg[2u * s] = qn_bf16x2(f[0], f[1]);
            kreg[2u * s + 1u] = qn_bf16x2(f[2], f[3]);
            qn_ldg4(f[0], f[1], f[2], f[3], v_cache + base);
            vreg[2u * s] = qn_bf16x2(f[0], f[1]);
            vreg[2u * s + 1u] = qn_bf16x2(f[2], f[3]);
        } else {
            kreg[2u * s] = 0u;
            kreg[2u * s + 1u] = 0u;
            vreg[2u * s] = 0u;
            vreg[2u * s + 1u] = 0u;
        }
    }
}

__device__ __forceinline__ void qn_store_kv(unsigned *sK, unsigned *sV,
                                            const unsigned (&kreg)[2 * (QN_KT * QN_HD) / (4 * QN_BLOCK)],
                                            const unsigned (&vreg)[2 * (QN_KT * QN_HD) / (4 * QN_BLOCK)],
                                            unsigned tid) {
    constexpr unsigned chunks = QN_HD / 4;
    constexpr unsigned steps = (QN_KT * QN_HD) / (4u * QN_BLOCK);
#pragma unroll
    for (unsigned s = 0u; s < steps; ++s) {
        const unsigned c = tid + s * QN_BLOCK;
        const unsigned r = c / chunks;
        const unsigned hd = (c % chunks) * 4u;
        unsigned *dst = sK + r * (QN_HSTRIDE / 2) + (hd >> 1);
        dst[0] = kreg[2u * s];
        dst[1] = kreg[2u * s + 1u];
        dst = sV + r * (QN_HSTRIDE / 2) + (hd >> 1);
        dst[0] = vreg[2u * s];
        dst[1] = vreg[2u * s + 1u];
    }
}

// Q fragment cache: one m16k16 A fragment per k step of the warp's 128-deep half,
// read straight from the FP32 q buffer in the mma A layout (a0/a1 = the two row
// halves at k, a2/a3 = the same rows at k + 8) and rounded to BF16 once.
__device__ __forceinline__ void qn_load_q(unsigned (&qf)[QN_KSTEPS][4],
                                          const float *__restrict__ q,
                                          unsigned lane, unsigned row0, unsigned q_head,
                                          unsigned half, unsigned tokens) {
    const unsigned g = lane >> 2u;
    const unsigned t = lane & 3u;
    const unsigned r_lo = row0 + g;
    const unsigned r_hi = r_lo + 8u;
    const bool ok_lo = r_lo < tokens;
    const bool ok_hi = r_hi < tokens;
    const float *qlo = q + (size_t)r_lo * QN_QSTRIDE + q_head * QN_HD;
    const float *qhi = q + (size_t)r_hi * QN_QSTRIDE + q_head * QN_HD;
#pragma unroll
    for (unsigned s = 0u; s < QN_KSTEPS; ++s) {
        const unsigned hd = half * QN_HALF + s * 16u + 2u * t;
        float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
        float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
        if (ok_lo) {
            qn_ldg2(x0, x1, qlo + hd);
            qn_ldg2(x2, x3, qlo + hd + 8u);
        }
        if (ok_hi) {
            qn_ldg2(y0, y1, qhi + hd);
            qn_ldg2(y2, y3, qhi + hd + 8u);
        }
        qf[s][0] = qn_bf16x2(x0, x1);
        qf[s][1] = qn_bf16x2(y0, y1);
        qf[s][2] = qn_bf16x2(x2, x3);
        qf[s][3] = qn_bf16x2(y2, y3);
    }
}

// ---------------------------------------------------------------------------
// kernel
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(QN_BLOCK)
qwen_native_attn(const float *__restrict__ q,
                 const float *__restrict__ gate,
                 const float *__restrict__ k_cache,
                 const float *__restrict__ v_cache,
                 float *__restrict__ out,
                 const unsigned int tokens,
                 const float scale,
                 const unsigned int pos0,
                 const unsigned int kv_len,
                 const unsigned int capacity) {
    // BF16 K/V tiles (one key row each) + the score exchange buffer of the four
    // q heads of this kv head. 41.25 KiB of STATIC shared memory, i.e. inside the
    // 48 KiB per-block static limit of sm_80+ (the native-shader launch path
    // requests no dynamic shared memory).
    __shared__ unsigned qn_sK[QN_KT * (QN_HSTRIDE / 2)];
    __shared__ unsigned qn_sV[QN_KT * (QN_HSTRIDE / 2)];
    __shared__ float qn_sS[QN_GQA * QN_QT * QN_SSTRIDE];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5u;
    const unsigned g = lane >> 2u;
    const unsigned t = lane & 3u;
    const unsigned qh_local = warp >> 1u;
    const unsigned half = warp & 1u;
    const unsigned kv_head = blockIdx.y;
    const unsigned q_head = kv_head * QN_GQA + qh_local;
    const unsigned row0 = blockIdx.x * QN_QT;
    if (row0 >= tokens) { return; }// whole block out of range: uniform exit

    // Causal window of this tile (identical to the DSL tiled kernel): the last row
    // of the tile needs every key <= pos0 + row_last, and neither kv_len nor the
    // cache capacity may be exceeded.
    const unsigned n_keys = kv_len < capacity ? kv_len : capacity;
    const unsigned row_last = (row0 + QN_QT - 1u) < tokens ? (row0 + QN_QT - 1u) : (tokens - 1u);
    const unsigned key_end = (pos0 + row_last + 1u) < n_keys ? (pos0 + row_last + 1u) : n_keys;

    const unsigned row_lo = row0 + g;
    const unsigned row_hi = row_lo + 8u;
    const unsigned lim_lo = pos0 + row_lo;
    const unsigned lim_hi = pos0 + row_hi;

    // Register state: the warp's [16 x 128] FP32 output accumulator, the running
    // online-softmax statistics per row half, and the BF16 Q fragments.
    float acc[QN_NPV][4];
    float run_m[2];
    float run_l[2];
    unsigned qf[QN_KSTEPS][4];
    unsigned kreg[2 * (QN_KT * QN_HD) / (4 * QN_BLOCK)];
    unsigned vreg[2 * (QN_KT * QN_HD) / (4 * QN_BLOCK)];
#pragma unroll
    for (unsigned i = 0u; i < QN_NPV; ++i) {
#pragma unroll
        for (unsigned j = 0u; j < 4u; ++j) { acc[i][j] = 0.0f; }
    }
    run_m[0] = QN_FLOOR;
    run_m[1] = QN_FLOOR;
    run_l[0] = 0.0f;
    run_l[1] = 0.0f;
    qn_load_q(qf, q, lane, row0, q_head, half, tokens);

    float *const sS = qn_sS + (qh_local * QN_QT) * QN_SSTRIDE;
    unsigned *const sK = qn_sK;
    unsigned *const sV = qn_sV;

    // Prologue: stage the first K/V tile.
    qn_load_kv(kreg, vreg, k_cache, v_cache, tid, kv_head, 0u, key_end);
    qn_store_kv(sK, sV, kreg, vreg, tid);
    __syncthreads();

    for (unsigned k0 = 0u; k0 < key_end; k0 += QN_KT) {
        // ---- QK^T: the warp's 128-deep half of the score -------------------
        float sc[QN_NQK][4];
#pragma unroll
        for (unsigned j = 0u; j < QN_NQK; ++j) {
#pragma unroll
            for (unsigned i = 0u; i < 4u; ++i) { sc[j][i] = 0.0f; }
        }
        // K B-fragment rows are the 8 keys of one n8 fragment; the two matrices of
        // the x2 load are head-dim k and k + 8 of the same 16-wide k step, and the
        // whole read is offset by the warp's head-dim half (`half * 128` elements).
        // Missing that half term makes both warps of a pair read the SAME 128
        // shortcut of K while their Q fragments live in different halves, which
        // silently doubles every score contribution of the second half.
        const unsigned krow = (lane & 7u) * (QN_HSTRIDE / 2);
        const unsigned khalf = ((lane >> 3u) & 1u) * 4u;
        const unsigned kbase = qn_smem_u32(sK) + (krow + khalf + half * (QN_HALF / 2u)) * 4u;
#pragma unroll
        for (unsigned s = 0u; s < QN_KSTEPS; ++s) {
            const unsigned koff = kbase + (s * 16u) * 2u;
#pragma unroll
            for (unsigned j = 0u; j < QN_NQK; ++j) {
                unsigned bf[2];
                qn_ldmatrix_x2(bf, koff + j * (8u * (QN_HSTRIDE / 2) * 4u));
                qn_mma(sc[j], qf[s], bf);
            }
        }

        // ---- score exchange + online softmax -------------------------------
        // One shared [16 x 32] FP32 buffer per q head carries the sister warp's
        // partial sum and then the summed, scaled and masked score: half 1 publishes
        // its partial, half 0 adds its own (two barriers, no extra buffer).
        if (half == 1u) {
#pragma unroll
            for (unsigned j = 0u; j < QN_NQK; ++j) {
                float *dst = sS + g * QN_SSTRIDE + j * 8u + 2u * t;
                dst[0] = sc[j][0];
                dst[1] = sc[j][1];
                dst = sS + (g + 8u) * QN_SSTRIDE + j * 8u + 2u * t;
                dst[0] = sc[j][2];
                dst[1] = sc[j][3];
            }
        }
        __syncthreads();
        // The whole tile is visible for every row of the block in the common case
        // (the diagonal-crossing tile is the only one that needs the mask).
        const bool tile_masked = (k0 + QN_KT - 1u) > lim_lo;
        if (half == 0u) {
#pragma unroll
            for (unsigned j = 0u; j < QN_NQK; ++j) {
                const unsigned key = k0 + j * 8u + 2u * t;
                float *lo = sS + g * QN_SSTRIDE + j * 8u + 2u * t;
                float *hi = sS + (g + 8u) * QN_SSTRIDE + j * 8u + 2u * t;
                float s0 = (sc[j][0] + lo[0]) * scale;
                float s1 = (sc[j][1] + lo[1]) * scale;
                float s2 = (sc[j][2] + hi[0]) * scale;
                float s3 = (sc[j][3] + hi[1]) * scale;
                if (tile_masked) {
                    s0 = key <= lim_lo ? s0 : QN_FLOOR;
                    s1 = (key + 1u) <= lim_lo ? s1 : QN_FLOOR;
                    s2 = key <= lim_hi ? s2 : QN_FLOOR;
                    s3 = (key + 1u) <= lim_hi ? s3 : QN_FLOOR;
                }
                lo[0] = s0;
                lo[1] = s1;
                hi[0] = s2;
                hi[1] = s3;
            }
        }
        __syncthreads();

        // Both warps of the pair now read the identical summed score, so both run
        // the identical FP32 softmax and build the identical BF16 P fragments.
        float sm[QN_NQK][4];
        float tm0 = QN_FLOOR;
        float tm1 = QN_FLOOR;
#pragma unroll
        for (unsigned j = 0u; j < QN_NQK; ++j) {
            const float *lo = sS + g * QN_SSTRIDE + j * 8u + 2u * t;
            const float *hi = sS + (g + 8u) * QN_SSTRIDE + j * 8u + 2u * t;
            sm[j][0] = lo[0];
            sm[j][1] = lo[1];
            sm[j][2] = hi[0];
            sm[j][3] = hi[1];
            tm0 = qn_max2(tm0, qn_max2(sm[j][0], sm[j][1]));
            tm1 = qn_max2(tm1, qn_max2(sm[j][2], sm[j][3]));
        }
        tm0 = qn_quad_max(tm0);
        tm1 = qn_quad_max(tm1);
        const float m0 = qn_max2(run_m[0], tm0);
        const float m1 = qn_max2(run_m[1], tm1);
        const float c0 = qn_ex2((run_m[0] - m0) * QN_LOG2E);
        const float c1 = qn_ex2((run_m[1] - m1) * QN_LOG2E);
        run_m[0] = m0;
        run_m[1] = m1;
        float l0 = 0.0f;
        float l1 = 0.0f;
#pragma unroll
        for (unsigned j = 0u; j < QN_NQK; ++j) {
            const float p0 = qn_ex2((sm[j][0] - m0) * QN_LOG2E);
            const float p1 = qn_ex2((sm[j][1] - m0) * QN_LOG2E);
            const float p2 = qn_ex2((sm[j][2] - m1) * QN_LOG2E);
            const float p3 = qn_ex2((sm[j][3] - m1) * QN_LOG2E);
            l0 += p0 + p1;
            l1 += p2 + p3;
            sm[j][0] = p0;
            sm[j][1] = p1;
            sm[j][2] = p2;
            sm[j][3] = p3;
        }
        l0 = qn_quad_sum(l0);
        l1 = qn_quad_sum(l1);
        run_l[0] = run_l[0] * c0 + l0;
        run_l[1] = run_l[1] * c1 + l1;
        // ONE accumulator rescale per tile (FlashAttention-2), then P.V.
#pragma unroll
        for (unsigned i = 0u; i < QN_NPV; ++i) {
            acc[i][0] *= c0;
            acc[i][1] *= c0;
            acc[i][2] *= c1;
            acc[i][3] *= c1;
        }
        unsigned pa[QN_KPV][4];
#pragma unroll
        for (unsigned j = 0u; j < QN_KPV; ++j) {
            // P is the mma A operand: keys 0-7 of the k step in a0/a1, keys 8-15 in
            // a2/a3, exactly the D-fragment pairing of the score above.
            pa[j][0] = qn_bf16x2(sm[2u * j][0], sm[2u * j][1]);
            pa[j][1] = qn_bf16x2(sm[2u * j][2], sm[2u * j][3]);
            pa[j][2] = qn_bf16x2(sm[2u * j + 1u][0], sm[2u * j + 1u][1]);
            pa[j][3] = qn_bf16x2(sm[2u * j + 1u][2], sm[2u * j + 1u][3]);
        }

        // ---- stage the next K/V tile through registers ---------------------
        // Issued BEFORE the P.V so its global latency is covered by mma work; the
        // shared-memory write happens after the barrier below.
        const unsigned next_k0 = k0 + QN_KT;
        const bool more = next_k0 < key_end;
        if (more) {
            qn_load_kv(kreg, vreg, k_cache, v_cache, tid, kv_head, next_k0, key_end);
        }

        // ---- P.V: acc += P * V over the tile, in this warp's 128 channels -----
        // V is read with ldmatrix.trans because the P.V reduction axis is the key
        // index, which is the strided axis of the [key][head_dim] tile.
        const unsigned vcol = half * QN_HALF;
        const unsigned vbase = qn_smem_u32(sV) + ((lane & 15u) * (QN_HSTRIDE / 2)) * 4u;
#pragma unroll
        for (unsigned j = 0u; j < QN_KPV; ++j) {
            const unsigned voff = vbase + (j * 16u) * (QN_HSTRIDE / 2) * 4u;
#pragma unroll
            for (unsigned i = 0u; i < QN_NPV; ++i) {
                unsigned bf[2];
                qn_ldmatrix_x2_trans(bf, voff + (vcol + i * 8u) * 2u);
                qn_mma(acc[i], pa[j], bf);
            }
        }

        __syncthreads();// sK/sV/QN_sS fully consumed by every warp
        if (more) {
            qn_store_kv(sK, sV, kreg, vreg, tid);
            __syncthreads();
        }
    }

    // ---- epilogue: normalise, apply the sigmoid gate, store ---------------
    const float inv0 = 1.0f / run_l[0];
    const float inv1 = 1.0f / run_l[1];
    const bool ok_lo = row_lo < tokens;
    const bool ok_hi = row_hi < tokens;
#pragma unroll
    for (unsigned i = 0u; i < QN_NPV; ++i) {
        const unsigned hd = half * QN_HALF + i * 8u + 2u * t;
        if (ok_lo) {
            float g0, g1;
            const float *gp = gate + (size_t)row_lo * QN_QSTRIDE + q_head * QN_HD + hd;
            qn_ldg2(g0, g1, gp);
            qn_stg2(out + (size_t)row_lo * QN_QSTRIDE + q_head * QN_HD + hd,
                    acc[i][0] * inv0 * qn_sigmoid(g0),
                    acc[i][1] * inv0 * qn_sigmoid(g1));
        }
        if (ok_hi) {
            float g0, g1;
            const float *gp = gate + (size_t)row_hi * QN_QSTRIDE + q_head * QN_HD + hd;
            qn_ldg2(g0, g1, gp);
            qn_stg2(out + (size_t)row_hi * QN_QSTRIDE + q_head * QN_HD + hd,
                    acc[i][2] * inv1 * qn_sigmoid(g0),
                    acc[i][3] * inv1 * qn_sigmoid(g1));
        }
    }
}
)QNATTN";

}// namespace luisa::qwen::dsl

#endif// QWEN_NATIVE_ATTN_KERNEL_H
