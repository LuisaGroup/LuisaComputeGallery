// CUDA C++ source template of the Qwen3.5-4B tensor-core prefill GEMM
// (opt_plan/09 N1a). Compiled at run time by `NativeShaderExt`
// (`NativeShaderLanguage::CUDA_NVRTC`) on the CUDA backend; no other backend has
// this route, and nothing here is part of the DSL/AST pipeline.
//
// Contract of the kernel (identical to `make_qwen_gemm_kernel` /
// `make_qwen35_gemm_mt_kernel`):
//   x        : [M, K] FP32 activations, row-major
//   w        : [N, K] BF16 weights, row-per-output, packed 2 codes per uint32
//              (i.e. exactly `__nv_bfloat16 w[N][K]`, contiguous in K)
//   scales   : FP8 only (unused by the BF16 realization; the parameter keeps the
//              ABI identical so one launcher serves both)
//   residual : [M, N] FP32, added when QN_RESIDUAL is 1 (a dummy buffer otherwise)
//   y        : [M, N] FP32 out, row-major
//   M, N, K  : runtime geometry
//
// The template is the single source of truth for the device code: the unit has no
// #include at all (NVRTC compiles it standalone), the tiling is injected as the
// `@TOKEN@` placeholders below, and the standalone nvcc harness under
// `.kimix_cache` substitutes exactly the same tokens, so what is differentially
// tested there is the text that runs in the engine.
//
// Numerics: `x` is converted FP32 -> BF16 *at the shared-tile boundary* (the
// global read is FP32, the shared tile and the mma operands are BF16) and the
// accumulation is FP32 (`f32.bf16.bf16.f32`). Weights are already BF16, so the
// only rounding this kernel adds relative to the FP32-reference GEMM is that
// single BF16 rounding of the activation, i.e. <= 2^-9 relative per product with
// random signs, which is ~2e-3 relative on an output element. See
// docs/examples/example_qwen.md for the measured tolerance.
//
// Warp tile is fixed at 32 rows x 32 columns (2 m16 subtiles x 4 n8 subtiles);
// the block tile (QN_BM x QN_BN), the k step (QN_BK), the warp count and the
// thread count are the placeholders, and stage A holds exactly 16 floats per
// thread (static_assert'ed below), so the global->register prefetch of stage A
// overlaps the mma of the current stage while the weight tile travels with
// cp.async into the other shared buffer.
#ifndef QWEN_NATIVE_GEMM_KERNEL_H
#define QWEN_NATIVE_GEMM_KERNEL_H

namespace luisa::qwen::dsl {

// `@NAME@` tokens are replaced by qwen_native_gemm_source() (host) and by the
// standalone nvcc harness. Keep this a raw literal: the engine hands it to NVRTC
// verbatim after substitution, and the CUDA backend's reflection parser reads
// the same text.
inline constexpr char qwen_native_gemm_cuda_template[] = R"QNCUDA(
// Qwen3.5-4B tensor-core prefill GEMM (BF16 weights, FP32 in/out).
// GENERATED from src/qwen/qwen/native_gemm_kernel.h - do not edit.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#error "qwen native prefill GEMM needs sm_80 or newer: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
#endif

#define QN_BM @BM@
#define QN_BN @BN@
#define QN_BK @BK@
#define QN_WARPS @WARPS@
#define QN_BLOCK @BLOCK@
#define QN_RESIDUAL @RESIDUAL@
#define QN_FP8 @FP8@
// FP8 E4M3 quantization group of the packed scales (64 in this model); unused by
// the BF16 realization, baked in so the scale index is a constant shift.
#define QN_GROUP @GROUP@
#define QN_NW (QN_BN / 32)
#define QN_MW (QN_WARPS / QN_NW)
// One shared row of the A or B tile: QN_BK/2 packed BF16 words plus 4 words of
// padding. The padding keeps the 16-byte alignment ldmatrix requires (4 words =
// 16 B) and spreads consecutive rows over the 32 banks.
#define QN_STRIDE (QN_BK / 2 + 4)
// float4 chunks per shared A row, and float4 chunks per thread per stage.
#define QN_A_CHUNKS (QN_BK / 4)
#define QN_A_STEPS (QN_BM * QN_BK / (4 * QN_BLOCK))
// 16-byte weight chunks per shared B row (8 BF16 each).
#define QN_B_CHUNKS (QN_BK / 8)
// FP8: packed words per shared B row (4 codes each) and per thread per stage. The
// E4M3 variant cannot use cp.async (the codes must be widened and scaled on the
// way into shared), so it prefetches like stage A does, through registers.
#define QN_B_WORDS (QN_BK / 4)
#define QN_B_STEPS (QN_BN * QN_B_WORDS / QN_BLOCK)

static_assert(QN_BK == 32, "the shared tile layout assumes a 32-element k step");
static_assert(QN_BM % 32 == 0, "block rows must be a multiple of the 32-row warp tile");
static_assert(QN_BN % 32 == 0, "block columns must be a multiple of the 32-column warp tile");
static_assert(QN_WARPS % QN_NW == 0, "the warp grid must tile the block");
static_assert(QN_BM / QN_MW == 32, "the warp tile is exactly 32 rows");
static_assert(QN_A_STEPS * 4 == 16, "stage A must be 16 floats per thread");
static_assert((QN_BN * QN_B_CHUNKS) % QN_BLOCK == 0, "stage B (16 B) must tile the block");
static_assert((QN_BN * (QN_BK / 2)) % QN_BLOCK == 0, "stage B (4 B) must tile the block");
static_assert(QN_BN * QN_B_WORDS % QN_BLOCK == 0, "stage B (FP8) must tile the block");
static_assert(QN_GROUP % 4 == 0, "one packed FP8 word must never straddle a scale group");
static_assert(QN_BLOCK == QN_WARPS * 32, "one 32-lane warp per warp tile");

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

__device__ __forceinline__ void qn_cpasync16(unsigned dst, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void qn_cpasync4(unsigned dst, const void *src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void qn_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until every cp.async group issued so far has landed. Exactly one group is
// outstanding at this point (the current stage), so this is the whole pipeline.
__device__ __forceinline__ void qn_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ unsigned qn_bf16x2(float lo, float hi) {
    unsigned short a, b;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(a) : "f"(lo));
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(b) : "f"(hi));
    return (unsigned)a | ((unsigned)b << 16);
}

__device__ __forceinline__ float qn_bits_to_f32(unsigned bits) {
    float f;
    asm("mov.b32 %0, %1;" : "=f"(f) : "r"(bits));
    return f;
}

// E4M3 FN -> FP32, bit-identical to qwen_fp8_e4m3_to_f32_branchless() in
// dsl_ops.cpp (which is itself verified against the reference decoder for all 256
// codes): normals are (s<<31)|((e+120)<<23)|(m<<20); a subnormal magnitude is
// (2^-6 + m*2^-9) - 2^-6, exact in FP32; S.1111.111 is NaN.
__device__ __forceinline__ float qn_e4m3(unsigned code) {
    const unsigned sign_bits = (code >> 7u) << 31u;
    const unsigned exp = (code >> 3u) & 0xFu;
    const unsigned mant = code & 0x7u;
    const float magic = qn_bits_to_f32(0x3C800000u);
    const float mag = qn_bits_to_f32(0x3C800000u | (mant << 20u)) - magic;
    const float sub = (code & 0x80u) ? -mag : mag;
    const float normal = qn_bits_to_f32(sign_bits | ((exp + 120u) << 23u) | (mant << 20u));
    const float finite = (exp == 0u) ? sub : normal;
    return (exp == 0xFu && mant == 0x7u) ? qn_bits_to_f32(sign_bits | 0x7FC00000u) : finite;
}

__device__ __forceinline__ void qn_ldmatrix_x4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void qn_ldmatrix_x2(unsigned (&r)[2], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(addr));
}

// D += A * B with A[16,16] row-major, B[16,8] "col" (k-major within an output
// column), FP32 accumulate.
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
// A: x[row0 .. row0+QN_BM) x [k0, k0+QN_BK) FP32 -> BF16 shared tile. This is the
// only place the activation is converted; the shared tile and every mma operand
// are BF16 from here on. `a4` says the rows of x are 16-byte aligned, i.e. K % 4
// == 0, so a float4 load covers a whole chunk.
__device__ __forceinline__ void qn_load_a_regs(float *regs, const float *__restrict__ x,
                                               unsigned tid, unsigned row0, unsigned k0,
                                               unsigned M, unsigned K, bool a4) {
    if (a4) {
#pragma unroll
        for (unsigned s = 0u; s < QN_A_STEPS; ++s) {
            const unsigned c = tid + s * QN_BLOCK;
            const unsigned r = c / QN_A_CHUNKS;
            const unsigned c4 = c % QN_A_CHUNKS;
            const unsigned m = row0 + r;
            const unsigned k = k0 + c4 * 4u;
            float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
            if (m < M && k + 4u <= K) {
                const float *p = x + (size_t)m * K + k;
                asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                             : "=f"(f0), "=f"(f1), "=f"(f2), "=f"(f3)
                             : "l"(p));
            } else if (m < M && k < K) {
                const float *p = x + (size_t)m * K + k;
                f0 = p[0];
                if (k + 1u < K) { f1 = p[1]; }
                if (k + 2u < K) { f2 = p[2]; }
                if (k + 3u < K) { f3 = p[3]; }
            }
            regs[4u * s + 0u] = f0;
            regs[4u * s + 1u] = f1;
            regs[4u * s + 2u] = f2;
            regs[4u * s + 3u] = f3;
        }
    } else {
#pragma unroll
        for (unsigned s = 0u; s < QN_A_STEPS; ++s) {
            const unsigned c = tid + s * QN_BLOCK;
            const unsigned r = c / QN_A_CHUNKS;
            const unsigned c4 = c % QN_A_CHUNKS;
            const unsigned m = row0 + r;
            const unsigned k = k0 + c4 * 4u;
            float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (m < M) {
                const float *p = x + (size_t)m * K + k;
#pragma unroll
                for (unsigned j = 0u; j < 4u; ++j) {
                    if (k + j < K) { f[j] = p[j]; }
                }
            }
#pragma unroll
            for (unsigned j = 0u; j < 4u; ++j) { regs[4u * s + j] = f[j]; }
        }
    }
}

// Stage A from the register cache (already in flight during the previous stage's
// mma work) into the BF16 shared tile.
__device__ __forceinline__ void qn_store_a_regs(unsigned *sA, const float *regs, unsigned tid) {
#pragma unroll
    for (unsigned s = 0u; s < QN_A_STEPS; ++s) {
        const unsigned c = tid + s * QN_BLOCK;
        const unsigned r = c / QN_A_CHUNKS;
        const unsigned c4 = c % QN_A_CHUNKS;
        unsigned *dst = sA + r * QN_STRIDE + c4 * 2u;
        dst[0] = qn_bf16x2(regs[4u * s + 0u], regs[4u * s + 1u]);
        dst[1] = qn_bf16x2(regs[4u * s + 2u], regs[4u * s + 3u]);
    }
}

// B (BF16): w[n0 .. n0+QN_BN) x [k0, k0+QN_BK) is contiguous in k, so a whole
// 16-byte chunk (8 BF16 codes) or a 4-byte word (2 codes) is one cp.async. `b16`
// says the weight rows are 16-byte aligned, i.e. K % 8 == 0; otherwise the 4-byte
// form is used, which any even K supports. Out-of-range rows/columns are zero
// filled: a chunk is either fully inside K (both offsets are even and the row
// stride is even) or fully outside.
__device__ __forceinline__ void qn_stage_b(unsigned *sB, const unsigned int *__restrict__ w,
                                           unsigned tid, unsigned n0, unsigned k0,
                                           unsigned N, unsigned K, bool b16) {
    const unsigned words_per_row = K >> 1u;
    if (b16) {
        constexpr unsigned steps = QN_BN * QN_B_CHUNKS / QN_BLOCK;
#pragma unroll
        for (unsigned s = 0u; s < steps; ++s) {
            const unsigned c = tid + s * QN_BLOCK;
            const unsigned r = c / QN_B_CHUNKS;
            const unsigned ck = c % QN_B_CHUNKS;
            const unsigned n = n0 + r;
            const unsigned k = k0 + ck * 8u;
            unsigned *dst = sB + r * QN_STRIDE + ck * 4u;
            if (n < N && k + 8u <= K) {
                qn_cpasync16(qn_smem_u32(dst), w + (size_t)n * words_per_row + (k >> 1u));
            } else {
                dst[0] = 0u;
                dst[1] = 0u;
                dst[2] = 0u;
                dst[3] = 0u;
            }
        }
    } else {
        constexpr unsigned steps = QN_BN * (QN_BK / 2) / QN_BLOCK;
#pragma unroll
        for (unsigned s = 0u; s < steps; ++s) {
            const unsigned c = tid + s * QN_BLOCK;
            const unsigned r = c / (QN_BK / 2);
            const unsigned ck = c % (QN_BK / 2);
            const unsigned n = n0 + r;
            const unsigned k = k0 + ck * 2u;
            unsigned *dst = sB + r * QN_STRIDE + ck;
            if (n < N && k + 2u <= K) {
                qn_cpasync4(qn_smem_u32(dst), w + (size_t)n * words_per_row + (k >> 1u));
            } else {
                dst[0] = 0u;
            }
        }
    }
}

#if QN_FP8
// B (FP8 E4M3): the packed codes and their per-group scales are read into
// REGISTERS one stage ahead (like stage A), widened to BF16 and scaled on the way
// into the shared tile. Nothing can go through cp.async here: the shared tile is
// BF16 while the buffer holds 4-bit-exponent bytes, and the scale is a per-group
// value. `K % 4 == 0` is a host-side precondition of this variant, so a packed word
// (4 codes) never straddles a row end, and `QN_GROUP % 4 == 0` guarantees it never
// straddles a scale group either - one scale load per word.
__device__ __forceinline__ void qn_load_b_fp8_regs(unsigned (&words)[QN_B_STEPS],
                                                   float (&scales_out)[QN_B_STEPS],
                                                   const unsigned int *__restrict__ w,
                                                   const float *__restrict__ scales,
                                                   unsigned tid, unsigned n0, unsigned k0,
                                                   unsigned N, unsigned K) {
    const unsigned words_per_row = K >> 2u;
    const unsigned groups_per_row = (K + (QN_GROUP - 1u)) / QN_GROUP;
#pragma unroll
    for (unsigned s = 0u; s < QN_B_STEPS; ++s) {
        const unsigned c = tid + s * QN_BLOCK;
        const unsigned r = c / QN_B_WORDS;
        const unsigned cw = c % QN_B_WORDS;
        const unsigned n = n0 + r;
        const unsigned k = k0 + cw * 4u;
        if (n < N && k + 4u <= K) {
            words[s] = w[(size_t)n * words_per_row + (k >> 2u)];
            scales_out[s] = scales[(size_t)n * groups_per_row + (k / QN_GROUP)];
        } else {
            words[s] = 0u;
            scales_out[s] = 0.0f;
        }
    }
}

__device__ __forceinline__ void qn_store_b_fp8_regs(unsigned *sB,
                                                    const unsigned (&words)[QN_B_STEPS],
                                                    const float (&scales_in)[QN_B_STEPS],
                                                    unsigned tid) {
#pragma unroll
    for (unsigned s = 0u; s < QN_B_STEPS; ++s) {
        const unsigned c = tid + s * QN_BLOCK;
        const unsigned r = c / QN_B_WORDS;
        const unsigned cw = c % QN_B_WORDS;
        const unsigned word = words[s];
        const float scale = scales_in[s];
        const float v0 = qn_e4m3(word & 0xFFu) * scale;
        const float v1 = qn_e4m3((word >> 8u) & 0xFFu) * scale;
        const float v2 = qn_e4m3((word >> 16u) & 0xFFu) * scale;
        const float v3 = qn_e4m3((word >> 24u) & 0xFFu) * scale;
        unsigned *dst = sB + r * QN_STRIDE + cw * 2u;
        dst[0] = qn_bf16x2(v0, v1);
        dst[1] = qn_bf16x2(v2, v3);
    }
}
#endif// QN_FP8

// ---------------------------------------------------------------------------
// kernel
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(QN_BLOCK)
qwen_native_gemm_bf16(const float *__restrict__ x,
                      const unsigned int *__restrict__ w,
                      const float *__restrict__ scales,
                      const float *__restrict__ residual,
                      float *__restrict__ y,
                      const unsigned int M,
                      const unsigned int N,
                      const unsigned int K) {
    __shared__ unsigned qn_sA[2][QN_BM * QN_STRIDE];
    __shared__ unsigned qn_sB[2][QN_BN * QN_STRIDE];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5u;
    const unsigned warp_m = warp / QN_NW;
    const unsigned warp_n = warp % QN_NW;

    const unsigned row0 = blockIdx.x * QN_BM;
    const unsigned col0 = blockIdx.y * QN_BN;
    const unsigned wrow0 = row0 + warp_m * 32u;
    const unsigned wcol0 = col0 + warp_n * 32u;
    // A warp whose whole 32x32 tile is outside the output contributes nothing; it
    // still takes part in the staging and in every barrier.
    const bool active = (wrow0 < M) && (wcol0 < N);

    const unsigned kt = (K + (QN_BK - 1u)) / QN_BK;
    const bool a4 = (K & 3u) == 0u;
#if !QN_FP8
    // 16-byte-copyable weight rows (K % 8 == 0). The FP8 arm widens through
    // registers and never issues a cp.async for B, so it has no such flag.
    const bool b16 = (K & 7u) == 0u;
#endif

    float regs[16];
    float acc[2][4][4];
#if QN_FP8
    unsigned bregs[QN_B_STEPS];
    float bscales[QN_B_STEPS];
#endif
#pragma unroll
    for (unsigned mi = 0u; mi < 2u; ++mi) {
#pragma unroll
        for (unsigned ni = 0u; ni < 4u; ++ni) {
#pragma unroll
            for (unsigned j = 0u; j < 4u; ++j) { acc[mi][ni][j] = 0.0f; }
        }
    }

    // Stage 0: x through registers (converted on the way into shared), weights
    // straight through cp.async.
    qn_load_a_regs(regs, x, tid, row0, 0u, M, K, a4);
#if QN_FP8
    qn_load_b_fp8_regs(bregs, bscales, w, scales, tid, col0, 0u, N, K);
#else
    qn_stage_b(qn_sB[0], w, tid, col0, 0u, N, K, b16);
#endif
    qn_store_a_regs(qn_sA[0], regs, tid);
#if QN_FP8
    qn_store_b_fp8_regs(qn_sB[0], bregs, bscales, tid);
#endif
    qn_commit();

    for (unsigned kb = 0u; kb < kt; ++kb) {
        const unsigned cur = kb & 1u;
        const unsigned nxt = cur ^ 1u;
        qn_wait_all();
        __syncthreads();

        const bool more = (kb + 1u < kt);
        if (more) {
            // Stage kb+1: the x read goes to registers now (so its latency is
            // covered by the mma below), the weight tile goes through cp.async.
            qn_load_a_regs(regs, x, tid, row0, (kb + 1u) * QN_BK, M, K, a4);
#if QN_FP8
            qn_load_b_fp8_regs(bregs, bscales, w, scales, tid, col0, (kb + 1u) * QN_BK, N, K);
#else
            qn_stage_b(qn_sB[nxt], w, tid, col0, (kb + 1u) * QN_BK, N, K, b16);
#endif
            qn_commit();
        }

        if (active) {
            const unsigned *sA = qn_sA[cur];
            const unsigned *sB = qn_sB[cur];
            const unsigned rr = lane & 7u;
            const unsigned quad = lane >> 3u;
            const unsigned Half = (quad >> 1u) << 3u;
            const unsigned bhalf = (lane >> 3u) & 1u;
#pragma unroll
            for (unsigned kk = 0u; kk < QN_BK; kk += 16u) {
                unsigned af[2][4];
#pragma unroll
                for (unsigned mi = 0u; mi < 2u; ++mi) {
                    // ldmatrix.x4: four 8x8 quadrants -> a0,a1 (k 0..7) and
                    // a2,a3 (k 8..15) of one 16x16 A fragment.
                    const unsigned row = warp_m * 32u + mi * 16u + ((quad & 1u) << 3u) + rr;
                    const unsigned col = kk + Half;
                    qn_ldmatrix_x4(af[mi], qn_smem_u32(sA + row * QN_STRIDE + (col >> 1u)));
                }
                unsigned bf[4][2];
#pragma unroll
                for (unsigned ni = 0u; ni < 4u; ++ni) {
                    // ldmatrix.x2: the two 8x8 k halves of one 8-column B fragment.
                    const unsigned row = warp_n * 32u + ni * 8u + rr;
                    const unsigned col = kk + (bhalf << 3u);
                    qn_ldmatrix_x2(bf[ni], qn_smem_u32(sB + row * QN_STRIDE + (col >> 1u)));
                }
#pragma unroll
                for (unsigned mi = 0u; mi < 2u; ++mi) {
#pragma unroll
                    for (unsigned ni = 0u; ni < 4u; ++ni) {
                        qn_mma(acc[mi][ni], af[mi], bf[ni]);
                    }
                }
            }
        }

        if (more) {
            qn_store_a_regs(qn_sA[nxt], regs, tid);
#if QN_FP8
            qn_store_b_fp8_regs(qn_sB[nxt], bregs, bscales, tid);
#endif
        }
    }

    // Epilogue: mma D fragment -> [M, N] FP32 (plus the optional residual).
    const unsigned g = lane >> 2u;
    const unsigned tg = lane & 3u;
#pragma unroll
    for (unsigned mi = 0u; mi < 2u; ++mi) {
        const unsigned r_lo = wrow0 + mi * 16u + g;
        const unsigned r_hi = r_lo + 8u;
#pragma unroll
        for (unsigned ni = 0u; ni < 4u; ++ni) {
            const unsigned c = wcol0 + ni * 8u + tg * 2u;
            if (c < N) {
                if (r_lo < M) {
                    float v0 = acc[mi][ni][0];
                    float v1 = acc[mi][ni][1];
                    if (QN_RESIDUAL) {
                        v0 += residual[(size_t)r_lo * N + c];
                        if (c + 1u < N) { v1 += residual[(size_t)r_lo * N + c + 1u]; }
                    }
                    float *p = y + (size_t)r_lo * N + c;
                    p[0] = v0;
                    if (c + 1u < N) { p[1] = v1; }
                }
                if (r_hi < M) {
                    float v2 = acc[mi][ni][2];
                    float v3 = acc[mi][ni][3];
                    if (QN_RESIDUAL) {
                        v2 += residual[(size_t)r_hi * N + c];
                        if (c + 1u < N) { v3 += residual[(size_t)r_hi * N + c + 1u]; }
                    }
                    float *p = y + (size_t)r_hi * N + c;
                    p[0] = v2;
                    if (c + 1u < N) { p[1] = v3; }
                }
            }
        }
    }
}
)QNCUDA";

}// namespace luisa::qwen::dsl

#endif// QWEN_NATIVE_GEMM_KERNEL_H
