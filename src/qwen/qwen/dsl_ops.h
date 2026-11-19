#pragma once
// Qwen3.5-4B text-only inference -- ordinary LuisaCompute DSL operators.
// Self-contained, device-free kernel factories + host planning/validation +
// independent CPU oracles + a device-free self-test entry. Namespace
// luisa::qwen::dsl. NO native shaders, NO Tile, NO executor/model/weights
// dependencies. Everything here compiles and runs without a Device.
//
// ---------------------------------------------------------------------------
// WEIGHT STORAGE ABI (matches .kimix_cache/qwen_implementation_contract.md)
// ---------------------------------------------------------------------------
// Two explicitly selectable packed weight formats; one is NEVER silently
// substituted for the other:
//   bf16       : Buffer<uint32_t> of K*N/2 words. Two BF16 code units per
//                little-endian word, LOW element first. Raw GGUF BF16 bytes.
//   fp8_e4m3   : Buffer<uint32_t> of ceil(K*N/4) words. Four E4M3 bytes per
//                little-endian word, LOW element first, plus FP32 scales:
//                one scale per contiguous group of `group` K elements per
//                output row (canonical group = 64). Dequantized weight =
//                fp8(code) * scale[row * ceil(K/group) + k/group].
//                E4M3 decoding is exact (finite codes incl. subnormals;
//                S.1111.111 is NaN per the OCP FN convention).
// Matrices are stored ROW-PER-OUTPUT, K CONTIGUOUS: W[n*K + k] is the weight
// of output n, input k (GGUF [K,N] payload row stride K). y[m,n] =
// sum_k x[m,k] * W[n,k]. Activations, accumulators, norms, softmax, gates and
// GDN state are FP32 everywhere; bf16/fp8 denotes WEIGHT STORAGE only.
//
// ---------------------------------------------------------------------------
// OPERATOR INVENTORY (each a distinct, explicitly named factory)
// ---------------------------------------------------------------------------
// embedding gather (packed rows, no dequant upload)
  // rmsnorm with DIRECT pre-added GGUF weights (converter folds the HF
  //   zero-centered 1+w offline) + optional residual add / residual snapshot
  //   (three epilogues: plain / residual / snapshot)
// packed GEMV (warp-per-output, single token / MTP verify small M)
// packed GEMM prefill (one thread per output, FP32 accumulation)
// SwiGLU over separate gate/up buffers or one fused [M,2F] projection
  // elementwise residual add / zero fill / copy (f32, u32, i32)
  // attention: fused Q/gate split + per-head QK norm (direct pre-added weight)
  // + NeoX rotate-half partial RoPE (+ optional K/V cache append), standalone
  // partial RoPE,
//   KV append, causal GQA prefill (online softmax, sigmoid output gate),
//   single-query decode (block-cooperative, sigmoid output gate)
// GDN: packed-QKV depthwise conv4 + SiLU with ring history, beta/decay gates,
  // per-QK-head FP32 L2 norm (rsqrt(sum+eps)) with Q scaling (32 V heads <-
  // 16 QK heads, TILED GGUF pairing h % qk_heads),
//   FP32 delta-rule recurrent decode/prefill plus WY representation + UT
//   transform over serial 64-token prefill chunks (one block per V head),
//   gated RMS output norm (direct weight, norm BEFORE SiLU(z) gate)
// CPU oracles for every nontrivial operator (double accumulation) and
// dsl_ops_self_test(): device-free, deterministic, ragged-shape coverage.
//
// WY/UT chunk-64 GDN prefill is implemented and differentially tested against
// the recurrent oracle; dsl_ops_chunk_prefill_status() reports success.
// Remaining operator gaps (reported, never silently faked):
//   * FP16/INT4/Q8_0 weight storage, paged KV, split-KV decode reduction,
//     fused gate/up PREFILL GEMM epilogue (decode fuses it; prefill SwiGLU is
//     separate). Host sampling and NextN live in engine/speculative modules.
//
// ---------------------------------------------------------------------------
// LAUNCH / PLANNING CONTRACT
// ---------------------------------------------------------------------------
// All kernels use TOTAL-THREAD dispatch semantics. Every plan_* helper is
// overflow-checked (uint64 host math, fail-closed on the uint32 kernel index
// envelope) and returns a recoverable Result; zero work yields an empty plan
// that callers must skip. 1D plans flatten into a 2D grid beyond the portable
// 65535-group envelope; kernels recover the linear id via
// dispatch_y()*dispatch_size().x + dispatch_x(). 2D plans fold rows into x
// when rows would exceed the group envelope; kernels recover (row,col) with
// the plan's padded_cols passed as a kernel argument.
// Every kernel guards its tail (`id < count`, `row < rows & col < cols`) and
// never reads/writes out of bounds for a validated plan. Warp-collective
// kernels use block-uniform guards so all lanes of a participating warp enter
// together. Kernels that pin set_warp_size(32) must be dispatched with
// plan.warp_size (host checks device.compute_warp_size(); the factory itself
// stays device-free).
//
// CommandList compatibility: factories take only buffer/scalar arguments and
// capture nothing, so `stream << shader(buf.view(), ..., scalars).dispatch(...)`
// composes into CommandList batches; no factory touches a Device or Stream.
#include <cstdint>

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/syntax.h>

#include "ocr/status.h"// shared recoverable Status/Result (same as qwen/common.h)

namespace luisa::qwen::dsl {

// ---------------------------------------------------------------------------
// Recoverable status / result: the SAME contract as qwen/common.h and the OCR
// engine (ocr/status.h). No separate error type is introduced.
// ---------------------------------------------------------------------------
using ocr::Result;
using ocr::Status;
using ocr::StatusCode;

// ---------------------------------------------------------------------------
// Model geometry (Qwen3.5-4B, text trunk; host constants).
// ---------------------------------------------------------------------------
struct QwenDslConfig {
    uint32_t hidden{2560};
    uint32_t ffn{9216};
    uint32_t vocab{248320};
    // full attention
    uint32_t q_heads{16};
    uint32_t kv_heads{4};
    uint32_t attn_head_dim{256};
    uint32_t rope_dim{64};      // partial rotary prefix (NeoX rotate-half pairs)
    float rope_theta{1.0e7f};
    // gated delta net
    uint32_t gdn_qk_heads{16};
    uint32_t gdn_v_heads{32};   // tiled GGUF order: V head h uses QK head h % qk_heads
    uint32_t gdn_head_dim{128};
    uint32_t conv_kernel{4};
    // numerics
    float norm_eps{1.0e-6f};
    float l2_eps{1.0e-6f};
    float softplus_threshold{20.0f};
    uint32_t fp8_group{64};     // canonical FP8 scale group along K

    [[nodiscard]] uint32_t attn_q_proj_n() const noexcept { return 2u * q_heads * attn_head_dim; }// [Q|gate] per head
    [[nodiscard]] uint32_t attn_kv_n() const noexcept { return kv_heads * attn_head_dim; }
    [[nodiscard]] uint32_t attn_o_k() const noexcept { return q_heads * attn_head_dim; }
    [[nodiscard]] uint32_t gdn_qk_n() const noexcept { return gdn_qk_heads * gdn_head_dim; }
    [[nodiscard]] uint32_t gdn_v_n() const noexcept { return gdn_v_heads * gdn_head_dim; }
    [[nodiscard]] uint32_t gdn_qkv_n() const noexcept { return 2u * gdn_qk_n() + gdn_v_n(); }
    [[nodiscard]] float attn_scale() const noexcept;// 1/sqrt(head_dim)
    [[nodiscard]] float gdn_q_scale() const noexcept;// 1/sqrt(head_dim)
};

// Host-side structural validation (head ratios, even dims, uint32 envelope of
// the fixed per-layer shapes). Recoverable; never throws.
[[nodiscard]] Status validate_qwen_config(const QwenDslConfig &cfg) noexcept;

// ---------------------------------------------------------------------------
// Weight format selection (explicit; no fallback).
// ---------------------------------------------------------------------------
enum class QwenWeightFormat : uint32_t {
    bf16 = 0,
    fp8_e4m3 = 1,
};
[[nodiscard]] luisa::string_view qwen_weight_format_name(QwenWeightFormat format) noexcept;

// Packed word counts for a K x N row-per-output matrix. Fails closed on
// overflow. bf16 requires an EVEN element count (the contract guarantees even
// K for this model); fp8 pads the last word with zero bytes.
[[nodiscard]] Result<uint64_t> packed_matrix_words(QwenWeightFormat format, uint64_t k, uint64_t n);
// FP8 scale element count: n * ceil(k/group).
[[nodiscard]] Result<uint64_t> fp8_scale_elements(uint64_t k, uint64_t n, uint32_t group);

// Host packing/conversion helpers (CPU; useful for tests and converters).
// bf16 round-to-nearest-even from FP32; fp8 E4M3 saturating quantization with
// scale = max_abs/448 per group (zero groups get scale 0, finite saturation).
[[nodiscard]] uint16_t host_f32_to_bf16_bits(float v) noexcept;
[[nodiscard]] float host_bf16_bits_to_f32(uint16_t bits) noexcept;
[[nodiscard]] uint8_t host_f32_to_fp8_e4m3_bits(float v) noexcept;// saturating, RNE
[[nodiscard]] float host_fp8_e4m3_bits_to_f32(uint8_t code) noexcept;
// Pack a dequantized FP32 [n,k] row-per-output matrix into the storage ABI.
// fp8 also fills `scales` (n*ceil(k/group)); quantization error is inherent
// and callers must report the conversion (contract: FP8 is not Q8_0).
[[nodiscard]] Status host_pack_matrix(QwenWeightFormat format, luisa::span<const float> w,
                                      uint64_t k, uint64_t n, uint32_t group,
                                      luisa::vector<uint32_t> &words,
                                      luisa::vector<float> &scales);
// Exact device-side decode parity helpers (same bit paths as the kernels).
[[nodiscard]] float host_decode_packed(QwenWeightFormat format, luisa::span<const uint32_t> words,
                                       luisa::span<const float> scales, uint64_t k,
                                       uint32_t group, uint64_t index) noexcept;

// ---------------------------------------------------------------------------
// Launch plans (TOTAL threads; overflow-checked; empty() == dispatch nothing).
// ---------------------------------------------------------------------------
inline constexpr uint32_t kQwenMaxGroupsPerDim = 65535u;

struct QwenLaunch1D {
    uint32_t total_x{0};
    uint32_t total_y{1};
    uint32_t block{0};
    uint32_t warp_size{0};// 0 = no pinned warp requirement; else exact device warp size
    uint64_t work_items{0};

    [[nodiscard]] bool empty() const noexcept { return work_items == 0u; }
    [[nodiscard]] bool is_2d() const noexcept { return total_y != 1u; }
};

struct QwenLaunch2D {
    uint32_t total_x{0};
    uint32_t total_y{0};
    uint32_t block_x{0};
    uint32_t block_y{0};
    uint64_t rows{0};
    uint64_t cols{0};
    uint32_t padded_cols{0};// in-kernel column stride for folded recovery
    bool folded{false};     // true: rows folded into x; use Kernel2D with padded_cols

    [[nodiscard]] bool empty() const noexcept { return rows == 0u || cols == 0u; }
};

// Generic elementwise plan: `count` items, blocks of `block` threads.
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_threads(uint64_t count, uint32_t block = 256u);

// Uniform dispatch helper: Kernel1D::dispatch takes a single total-x count,
// so a flattened plan (is_2d()) must be issued as a 1D launch of
// total_x * total_y threads. The kernels recover the linear id with
// dispatch_y()*dispatch_size().x + dispatch_x(), which stays correct because
// dispatch_size().y == 1 in that form. Use:
//   stream << shader(args...).dispatch(qwen_dispatch_threads(plan));
[[nodiscard]] inline uint32_t qwen_dispatch_threads(const QwenLaunch1D &plan) noexcept {
    return plan.total_x * plan.total_y;
}
// 2D kernels: dispatch(total_x, total_y) directly (folded or not).
// Generic 2D plan (one thread per (row,col)); folds rows into x past the
// group envelope. rows/cols must fit uint32 individually.
[[nodiscard]] Result<QwenLaunch2D> plan_qwen_threads_2d(uint64_t rows, uint64_t cols,
                                                         uint32_t block_x = 16u, uint32_t block_y = 16u);

// ---------------------------------------------------------------------------
// 1. Embedding gather. One thread per (token, channel): out[t,c] =
//    decode(table[id[t], c]). table is packed [vocab, hidden] (bf16 words or
//    fp8 words + per-row-group scales with the row stride ceil(hidden/group)).
// ---------------------------------------------------------------------------
using QwenEmbeddingKernel = compute::Kernel2D<void(
    compute::Buffer<uint32_t> /*ids*/,
    compute::Buffer<uint32_t> /*table packed [vocab,hidden]*/,
    compute::Buffer<float> /*scales (fp8; bind 1-elem dummy for bf16)*/,
    compute::Buffer<float> /*out [tokens,hidden]*/,
    uint32_t /*tokens*/, uint32_t /*hidden*/, uint32_t /*vocab*/,
    uint32_t /*group (fp8)*/, uint32_t /*padded_cols*/)>;
[[nodiscard]] QwenEmbeddingKernel make_qwen_embedding_kernel(QwenWeightFormat format) noexcept;
// ids[t] >= vocab is a caller error: validate host-side; the kernel skips
// out-of-range ids defensively (leaves the row untouched) instead of reading
// OOB. Plan rows=tokens, cols=hidden.
[[nodiscard]] Result<QwenLaunch2D> plan_qwen_embedding_dispatch(uint64_t tokens, uint64_t hidden);
[[nodiscard]] Status validate_qwen_embedding_inputs(luisa::span<const uint32_t> ids,
                                                    uint64_t hidden, uint64_t vocab);

// ---------------------------------------------------------------------------
// 2. Zero-centered RMSNorm (FP32): y = x * rsqrt(mean(x^2)+eps) * w_direct.
//    One 32-lane warp per row of `dim`; warp_active_sum reduction. Epilogues:
//      plain    : y = norm(x); bind 1-element dummies for residual AND sum_out.
//      residual : sum_out = x + residual, then y = norm(sum_out); residual
//                 and sum_out are full [rows,dim] buffers.
//      snapshot : sum_out = x (bit-exact copy written during the existing
//                 sum-of-squares pass), then y = norm(sum_out). Identical
//                 FP32 math and y values to plain (the copied values ARE the
//                 original x elements); additionally emits the pre-norm
//                 residual stream. This fuses residual_snapshot + input_norm
//                 into ONE launch: the executor's separate snapshot copy (one
//                 full activation read + one launch) disappears. Bind a
//                 1-element dummy for residual; sum_out is [rows,dim].
//    ALIASING CONTRACT: x, w, y and sum_out must be DISTINCT buffers (the
//    intended engine use binds separate activation/weight/output/snapshot
//    buffers). Aliasing is NOT supported: each lane writes sum_out[i] before
//    any lane re-reads it, but other lanes may not have read x[i] yet, so
//    sum_out aliasing x/y/w is a cross-lane race with undefined results.
//    rows = tokens (or tokens*heads for QK norm).
// ---------------------------------------------------------------------------
enum class QwenNormEpilogue : uint32_t {
    plain = 0,
    residual = 1,// sum_out = x + residual; y = norm(sum_out)
    snapshot = 2,// sum_out = x (exact copy); y = norm(sum_out); residual dummy
};
using QwenRmsNormKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x [rows,dim]*/,
    compute::Buffer<float> /*w [dim] pre-added GGUF weight (direct multiply)*/,
    compute::Buffer<float> /*residual [rows,dim] (dummy when plain/snapshot)*/,
    compute::Buffer<float> /*y [rows,dim]*/,
    compute::Buffer<float> /*sum_out [rows,dim] (dummy when plain)*/,
    uint32_t /*rows*/, uint32_t /*dim*/, float /*eps*/)>;
// rows must fit the plan's warp-count envelope; dim may be ragged (lanes
// guard their own elements; the warp reduction is block-uniform).
[[nodiscard]] QwenRmsNormKernel make_qwen_rmsnorm_kernel(QwenNormEpilogue epilogue) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_rmsnorm_dispatch(uint64_t rows, uint64_t dim);
// Wide-row realization used by the 2560-channel Qwen model. One full
// 128-thread block owns each row (instead of one active warp in a 128-thread
// block): warp reductions publish four shared partials, then every thread
// applies the normalized epilogue to its strided elements. Same ABI/epilogues.
[[nodiscard]] QwenRmsNormKernel make_qwen35_rmsnorm_kernel(
    QwenNormEpilogue epilogue, uint32_t block_size = 128u) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_rmsnorm_dispatch(
    uint64_t rows, uint64_t dim, uint32_t block_size = 128u);

// ---------------------------------------------------------------------------
// 3. Packed GEMV (single token; small M for MTP verify). y[m,n] =
//    sum_k x[m,k]*W[n,k] (+ residual[m,n] when epilogue=residual).
//    One 32-lane warp per output column n; FP32 accumulation +
//    warp_active_sum. M <= 4.
//    K partition: when K is divisible by the pack width (2 for bf16, 4 for
//    fp8) AND, for fp8, group % 4 == 0 (a quad never crosses a scale), each
//    lane owns whole packed words (ki = lane*pack, K stride 32*pack): one
//    word read/decode per pair/quad, one fp8 scale load per quad, ascending
//    in-word product order. Ragged/unsupported K/group uses the legacy
//    per-element strided-K loop in-kernel. The two paths partition K across
//    lanes differently, so warp-reduction rounding differs between them
//    (accumulation-order noise only; NOT bit-identical). Same ABI/epilogues.
// ---------------------------------------------------------------------------
inline constexpr uint32_t kQwenGemvMaxM = 4u;
enum class QwenMatmulEpilogue : uint32_t {
    none = 0,
    residual = 1,// y += residual (dummy 1-elem buffer when none)
};
using QwenGemvKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x [M,K]*/,
    compute::Buffer<uint32_t> /*W packed [N,K] row-per-output*/,
    compute::Buffer<float> /*scales (fp8; dummy for bf16)*/,
    compute::Buffer<float> /*residual [M,N] (dummy when none)*/,
    compute::Buffer<float> /*y [M,N]*/,
    uint32_t /*M*/, uint32_t /*N*/, uint32_t /*K*/, uint32_t /*group (fp8)*/)>;
[[nodiscard]] QwenGemvKernel make_qwen_gemv_kernel(QwenWeightFormat format,
                                                   QwenMatmulEpilogue epilogue) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gemv_dispatch(uint64_t m, uint64_t n, uint64_t k);
// Qwen3.5-4B decode-only M=1 realization. Four independent packed-word
// accumulators per lane expose memory-level parallelism; FP8 uses an exact
// 256-entry shared LUT instead of per-weight exponent normalization loops.
// The real K dimensions (2560/4096/9216) divide 128 packed words exactly, so
// the inner loop has no tail predicates. The generic factory above remains
// the ragged/M<=4 reference. Reduction order differs within FP32 tolerance.
[[nodiscard]] QwenGemvKernel make_qwen35_decode_gemv_kernel(
    QwenWeightFormat format, QwenMatmulEpilogue epilogue) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_decode_gemv_dispatch(
    uint64_t n, uint64_t k, QwenWeightFormat format);

// ---------------------------------------------------------------------------
// 4. Packed GEMM prefill. Same math/ABI as GEMV; one thread per (m,n) output,
//    serial FP32 dot over K (weights of one output row are contiguous).
//    Correctness-first baseline; a tiled/shared-memory variant is future work.
// ---------------------------------------------------------------------------
using QwenGemmKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*x [M,K]*/,
    compute::Buffer<uint32_t> /*W packed [N,K]*/,
    compute::Buffer<float> /*scales (fp8; dummy for bf16)*/,
    compute::Buffer<float> /*residual [M,N] (dummy when none)*/,
    compute::Buffer<float> /*y [M,N]*/,
    uint32_t /*M*/, uint32_t /*N*/, uint32_t /*K*/, uint32_t /*group (fp8)*/,
    uint32_t /*padded_cols*/)>;
[[nodiscard]] QwenGemmKernel make_qwen_gemm_kernel(QwenWeightFormat format,
                                                   QwenMatmulEpilogue epilogue) noexcept;
[[nodiscard]] Result<QwenLaunch2D> plan_qwen_gemm_dispatch(uint64_t m, uint64_t n, uint64_t k);

// ---------------------------------------------------------------------------
// 5. SwiGLU: out[i] = silu(gate[i]) * up[i].
//    split  : separate gate/up buffers, count elements each.
//    fused  : one [M, 2F] projection, row layout [gate F | up F]; out [M,F].
//    Two DISTINCT factories; one is never substituted for the other.
// ---------------------------------------------------------------------------
using QwenSwiGluSplitKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*gate*/, compute::Buffer<float> /*up*/,
    compute::Buffer<float> /*out*/, uint32_t /*count*/)>;
using QwenSwiGluFusedKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*proj [M,2F]*/, compute::Buffer<float> /*out [M,F]*/,
    uint32_t /*M*/, uint32_t /*F*/, uint32_t /*padded_cols*/)>;
[[nodiscard]] QwenSwiGluSplitKernel make_qwen_swiglu_split_kernel() noexcept;
[[nodiscard]] QwenSwiGluFusedKernel make_qwen_swiglu_fused_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_swiglu_dispatch(uint64_t count);
[[nodiscard]] Result<QwenLaunch2D> plan_qwen_swiglu_fused_dispatch(uint64_t m, uint64_t f);

// Decode-only gate/up projection + SwiGLU fusion. One warp owns one FFN
// channel, loads each activation pair/quad once, accumulates independent gate
// and up dots, and applies SiLU after the two warp reductions. This removes
// two launches and both full gate/up intermediate round trips per layer.
using QwenFfnDecodeKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x [K]*/,
    compute::Buffer<uint32_t> /*gate W [F,K]*/,
    compute::Buffer<float> /*gate scales*/,
    compute::Buffer<uint32_t> /*up W [F,K]*/,
    compute::Buffer<float> /*up scales*/,
    compute::Buffer<float> /*out [F]*/,
    uint32_t /*F*/, uint32_t /*K*/, uint32_t /*group*/)>;
[[nodiscard]] QwenFfnDecodeKernel make_qwen_ffn_decode_kernel(QwenWeightFormat format) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_ffn_decode_dispatch(
    uint64_t f, uint64_t k, QwenWeightFormat format, uint32_t group = 64u);

// ---------------------------------------------------------------------------
// 6. Elementwise utilities.
// ---------------------------------------------------------------------------
using QwenResidualAddKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*a*/, compute::Buffer<float> /*b*/,
    compute::Buffer<float> /*out*/, uint32_t /*count*/)>;
[[nodiscard]] QwenResidualAddKernel make_qwen_residual_add_kernel() noexcept;
using QwenZeroFillKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*out*/, uint32_t /*count*/)>;
[[nodiscard]] QwenZeroFillKernel make_qwen_zero_fill_kernel() noexcept;
// Elementwise FP32 copy: out[i] = src[i]. Added for the DSL executor, which
// must snapshot the residual stream (hidden -> residual) at each block entry;
// residual_add cannot express a pure copy (it always adds a second operand).
  using QwenCopyKernel = compute::Kernel1D<void(
      compute::Buffer<float> /*src*/, compute::Buffer<float> /*out*/, uint32_t /*count*/)>;
  [[nodiscard]] QwenCopyKernel make_qwen_copy_kernel() noexcept;
  // Scalar-parameter staging copy: out[i] = src[i], uint32 elements.
  // Rationale: host uploads via copy_from(span) record a BORROWED host pointer
  // that is only dereferenced at command-encoder commit time; a caller-local
  // staging vector can be destroyed before a queued CommandList is submitted
  // (dangling read -> nondeterministic garbage). The executor therefore stages
  // scalar inputs (token ids, RoPE positions) into persistent host members and
  // copies them into device buffers with this kernel instead of copy_from.
  using QwenCopyUKernel = compute::Kernel1D<void(
      compute::Buffer<uint32_t> /*src*/, compute::Buffer<uint32_t> /*out*/, uint32_t /*count*/)>;
  [[nodiscard]] QwenCopyUKernel make_qwen_copy_u_kernel() noexcept;
  using QwenCopyIKernel = compute::Kernel1D<void(
      compute::Buffer<int32_t> /*src*/, compute::Buffer<int32_t> /*out*/, uint32_t /*count*/)>;
  [[nodiscard]] QwenCopyIKernel make_qwen_copy_i_kernel() noexcept;

// Greedy full-vocabulary reduction for in-flight decode. One 256-thread block
// scans logits coalesced, preserves lowest-index ties (including +/-inf), and
// writes history[slot] and next_ids[0]. Any NaN produces vocab as a fail-closed
// sentinel; the host validates it at the bounded chunk synchronization.
using QwenGreedyArgmaxKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*logits [vocab]*/,
    compute::Buffer<uint32_t> /*next_ids [1]*/,
    compute::Buffer<uint32_t> /*history [generated slots]*/,
    uint32_t /*vocab*/, uint32_t /*slot*/)>;
[[nodiscard]] QwenGreedyArgmaxKernel make_qwen_greedy_argmax_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_greedy_argmax_dispatch(uint64_t vocab);
// Parallel engine realization: 256 blocks reduce disjoint strided vocab
// subsets, then one block reduces the 256 candidates. Two launches replace the
// reference single-block full-vocab scan; semantics and sentinel are unchanged.
inline constexpr uint32_t qwen_greedy_argmax_blocks = 256u;
using QwenGreedyArgmaxPartialKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*logits*/, compute::Buffer<float> /*values [256]*/,
    compute::Buffer<uint32_t> /*indices [256]*/, compute::Buffer<uint32_t> /*invalid [256]*/,
    uint32_t /*vocab*/)>;
using QwenGreedyArgmaxFinishKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*values [256]*/, compute::Buffer<uint32_t> /*indices [256]*/,
    compute::Buffer<uint32_t> /*invalid [256]*/, compute::Buffer<uint32_t> /*next_ids*/,
    compute::Buffer<uint32_t> /*history*/, uint32_t /*vocab*/, uint32_t /*slot*/)>;
[[nodiscard]] QwenGreedyArgmaxPartialKernel make_qwen_greedy_argmax_partial_kernel() noexcept;
[[nodiscard]] QwenGreedyArgmaxFinishKernel make_qwen_greedy_argmax_finish_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_greedy_argmax_partial_dispatch(uint64_t vocab);
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_greedy_argmax_finish_dispatch(uint64_t vocab);

// ---------------------------------------------------------------------------
// 7. Full attention.
//
// q_proj output layout (per token): q_heads * [Q(head_dim) | gate(head_dim)]
// interleaved PER HEAD (N = 2*q_heads*head_dim = 8192). k_proj/v_proj are
// plain [kv_heads*head_dim]. QK norm multiplies the PRE-ADDED GGUF weight
// DIRECTLY (the converter folds the HF zero-centered 1+w offline) per head
// over head_dim; RoPE rotates ONLY the first rope_dim elements with NeoX
// rotate-half pairs (i, i+rope_dim/2), inv_freq[i] = theta^(-2i/rope_dim),
// duplicated cos/sin across the halves; the tail [rope_dim, head_dim) passes
// through. Text mode: one position per token (the three mRoPE axes collapse
// to the same frequency table). Norm happens BEFORE RoPE.
// ---------------------------------------------------------------------------

// Fused split + QK norm + partial RoPE. One 32-lane warp per (token, head)
// row; warp covers head_dim with a strided loop and warp_active_sum for the
// norm. Writes q [T, q_heads, hd] (normed+roped), gate [T, q_heads, hd]
// (raw), k [T, kv_heads, hd] (normed+roped), v [T, kv_heads, hd] (raw).
// When append=true, k/v are ALSO written into the contiguous caches at
// absolute positions pos0+t (cache [capacity, kv_heads, hd]); bind dummies
// otherwise. positions[t] is the absolute text position (int32 >= 0).
using QwenAttnPrepareKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q_proj [T, 2*q_heads*hd]*/,
    compute::Buffer<float> /*k_proj [T, kv_heads*hd]*/,
    compute::Buffer<float> /*v_proj [T, kv_heads*hd]*/,
    compute::Buffer<float> /*q_norm_w [hd]*/,
    compute::Buffer<float> /*k_norm_w [hd]*/,
    compute::Buffer<int32_t> /*positions [T]*/,
    compute::Buffer<float> /*q out [T, q_heads, hd]*/,
    compute::Buffer<float> /*gate out [T, q_heads, hd]*/,
    compute::Buffer<float> /*k out [T, kv_heads, hd]*/,
    compute::Buffer<float> /*v out [T, kv_heads, hd]*/,
    compute::Buffer<float> /*k_cache (dummy when !append)*/,
    compute::Buffer<float> /*v_cache (dummy when !append)*/,
    uint32_t /*tokens*/, uint32_t /*q_heads*/, uint32_t /*kv_heads*/,
    uint32_t /*head_dim*/, uint32_t /*rope_dim*/, float /*theta*/, float /*eps*/,
    uint32_t /*pos0*/, uint32_t /*cache_capacity*/,
    uint32_t /*q_rows = tokens*q_heads*/, uint32_t /*k_rows = tokens*kv_heads*/,
    uint32_t /*v_rows = tokens*kv_heads*/)>;
[[nodiscard]] QwenAttnPrepareKernel make_qwen_attn_prepare_kernel(bool append_kv) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_prepare_dispatch(uint64_t tokens, const QwenDslConfig &cfg);
[[nodiscard]] Status validate_qwen_attn_prepare_inputs(uint64_t tokens, const QwenDslConfig &cfg,
                                                       uint64_t pos0, uint64_t cache_capacity, bool append_kv);

// Standalone NeoX rotate-half partial RoPE over [T, heads, hd] (one warp per
// row). Used when the executor keeps prepare and RoPE separate. In-place safe
// (x == out) because each lane owns its pair exclusively.
using QwenRopePartialKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x [T,heads,hd]*/,
    compute::Buffer<float> /*out [T,heads,hd]*/,
    compute::Buffer<int32_t> /*positions [T]*/,
    uint32_t /*tokens*/, uint32_t /*heads*/, uint32_t /*head_dim*/,
    uint32_t /*rope_dim*/, float /*theta*/, uint32_t /*rows = tokens*heads*/)>;
[[nodiscard]] QwenRopePartialKernel make_qwen_rope_partial_neox_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_rope_dispatch(uint64_t tokens, uint64_t heads);

// KV append: copy k/v [T, kv_heads, hd] into cache [capacity, kv_heads, hd]
// at pos0+t. One thread per (token, channel).
using QwenKvAppendKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*k [T,C]*/, compute::Buffer<float> /*v [T,C]*/,
    compute::Buffer<float> /*k_cache*/, compute::Buffer<float> /*v_cache*/,
    uint32_t /*tokens*/, uint32_t /*channels*/, uint32_t /*pos0*/,
    uint32_t /*capacity*/, uint32_t /*padded_cols*/)>;
[[nodiscard]] QwenKvAppendKernel make_qwen_kv_append_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch2D> plan_qwen_kv_append_dispatch(uint64_t tokens, uint64_t channels);

// Causal GQA prefill attention with sigmoid output gate. Queries are the T
// tokens at absolute positions pos0..pos0+T-1; keys/values are the cache
// [capacity, kv_heads, hd] with kv_len valid entries (kv_len >= pos0+T after
// append). Query t attends keys [0, pos0+t] (causal diagonal aligned to the
// sequence end). One thread per (token, q_head); single-pass online FP32
// softmax (running max/sum/output), GQA maps q_head -> q_head/(q_heads/kv_heads).
// out[t,h,:] = softmax(q.K^T * scale) . V * sigmoid(gate[t,h,:]).
using QwenAttnPrefillKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q [T,q_heads,hd]*/,
    compute::Buffer<float> /*gate [T,q_heads,hd]*/,
    compute::Buffer<float> /*k_cache [capacity,kv_heads,hd]*/,
    compute::Buffer<float> /*v_cache [capacity,kv_heads,hd]*/,
    compute::Buffer<float> /*out [T,q_heads,hd]*/,
    uint32_t /*tokens*/, uint32_t /*q_heads*/, uint32_t /*kv_heads*/,
    uint32_t /*head_dim*/, float /*scale*/, uint32_t /*pos0*/, uint32_t /*kv_len*/)>;
[[nodiscard]] QwenAttnPrefillKernel make_qwen_attn_prefill_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_prefill_dispatch(uint64_t tokens, uint64_t q_heads);
[[nodiscard]] Status validate_qwen_attn_prefill_inputs(uint64_t tokens, uint64_t q_heads, uint64_t kv_heads,
                                                       uint64_t head_dim, uint64_t pos0, uint64_t kv_len,
                                                       uint64_t capacity);

// Single-query decode attention. One BLOCK (128 threads, 4 warps) per q_head.
// Each warp runs an independent online FP32 softmax over its strided key
// subset (lane l handles dims l, l+32, ...; warp_active_sum for the score
// and running statistics), publishes (m, l, acc[d]) to shared memory, and
// thread 0 merges the four partials deterministically (empty subsets are
// skipped) and applies the sigmoid gate. q/gate [q_heads, hd]; cache
// [capacity, kv_heads, hd]; out [q_heads, hd]. kv_len >= 1 required.
// head_dim must be a multiple of 32 in [32,256] (shared tile bound).
using QwenAttnDecodeKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q [q_heads,hd]*/,
    compute::Buffer<float> /*gate [q_heads,hd]*/,
    compute::Buffer<float> /*k_cache*/, compute::Buffer<float> /*v_cache*/,
    compute::Buffer<float> /*out [q_heads,hd]*/,
    uint32_t /*q_heads*/, uint32_t /*kv_heads*/, uint32_t /*head_dim*/,
    float /*scale*/, uint32_t /*kv_len*/, uint32_t /*capacity*/)>;
// Pins warp size 32; host must check device.compute_warp_size() == 32.
[[nodiscard]] QwenAttnDecodeKernel make_qwen_attn_decode_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_decode_dispatch(uint64_t q_heads);
[[nodiscard]] Status validate_qwen_attn_decode_inputs(uint64_t q_heads, uint64_t kv_heads,
                                                      uint64_t head_dim, uint64_t kv_len, uint64_t capacity);

// ---------------------------------------------------------------------------
// 8. Gated DeltaNet.
//
// Packed QKV projection output layout per token: [Q(2048) | K(2048) | V(4096)]
// (16 QK heads * 128, 32 V heads * 128). The depthwise conv weight is the GGUF
// ssm_conv1d tensor [kernel=4, channels=8192] whose payload is CHANNEL-major:
// weight[c*kernel + j] is tap j of channel c.
// Ring history holds the last kernel-1 PRE-activation projected values per
// channel: ring[c, s], s = absolute_position % (kernel-1). Conv reads taps
// from the ring for positions < pos0 and from src for positions >= pos0,
// then pushes src into the ring AFTER the output (causal, race-free).
// Gates: beta = sigmoid(b); g = ssm_a[h] * softplus(a + dt_bias[h]) where the
// GGUF ssm_a is ALREADY -exp(A_log) (negative; folded offline by the
// converter), so g <= 0 without any runtime exp of ssm_a. Q/K are FP32
// L2-normalized per QK head as x*rsqrt(sum(x*x)+eps) (FLA/HF convention, eps
// INSIDE the rsqrt); Q additionally scales by 1/sqrt(128). GGUF V-side
// tensors are TILED (converter reorders grouped HF heads for ggml broadcast):
// V head h uses QK head h % qk_heads (no tensor replication). Recurrent state
// per V head: S[k,v] [128,128]
// FP32, layout state[h, k, v]. Delta rule per token:
//   S <- exp(g) * S;  err = v - S^T k;  S += beta * k err^T;  o = S^T q.
// Output norm: y = o * rsqrt(mean(o^2)+eps) * norm_w * silu(z)  (DIRECT
// weight, norm BEFORE gate; ssm_norm is the one GGUF norm the converter does
// NOT pre-add, and the generic RMSNorm above also multiplies directly
// because all other GGUF norms arrive pre-added).
// ---------------------------------------------------------------------------

// Conv4 + SiLU over the packed QKV channels with ring history (in/out).
// One thread per channel; tokens processed in causal order in-thread.
using QwenGdnConvKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*src [T,C] pre-activation qkv*/,
    compute::Buffer<float> /*weight [kernel,C] GGUF channel-major payload w[c*K+j]*/,
    compute::Buffer<float> /*ring [C,kernel-1] in/out*/,
    compute::Buffer<float> /*out [T,C]*/,
    uint32_t /*tokens*/, uint32_t /*channels*/, uint32_t /*kernel_size*/, uint32_t /*pos0*/)>;
[[nodiscard]] QwenGdnConvKernel make_qwen_gdn_conv_silu_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_conv_dispatch(uint64_t channels);
[[nodiscard]] Status validate_qwen_gdn_conv_inputs(uint64_t tokens, uint64_t channels,
                                                   uint64_t kernel_size, uint64_t pos0);

// beta/g gates. a/b [T, v_heads]; ssm_a/dt_bias [v_heads]; beta/g out [T, v_heads].
using QwenGdnGatesKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*a*/, compute::Buffer<float> /*b*/,
    compute::Buffer<float> /*ssm_a [H]*/, compute::Buffer<float> /*dt_bias [H]*/,
    compute::Buffer<float> /*beta out*/, compute::Buffer<float> /*g out*/,
    uint32_t /*tokens*/, uint32_t /*heads*/, float /*softplus_threshold*/)>;
[[nodiscard]] QwenGdnGatesKernel make_qwen_gdn_gates_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_gates_dispatch(uint64_t tokens, uint64_t heads);
// T=1 fusion of the alpha/beta packed GEMVs and their gate epilogues. One
// warp per head and gate kind (2*heads warps) writes beta or g directly.
using QwenGdnDecodeGatesKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x [K]*/,
    compute::Buffer<uint32_t> /*alpha W [H,K]*/, compute::Buffer<float> /*alpha scales*/,
    compute::Buffer<uint32_t> /*beta W [H,K]*/, compute::Buffer<float> /*beta scales*/,
    compute::Buffer<float> /*ssm_a [H]*/, compute::Buffer<float> /*dt_bias [H]*/,
    compute::Buffer<float> /*beta out [H]*/, compute::Buffer<float> /*g out [H]*/,
    uint32_t /*K*/, uint32_t /*heads*/, uint32_t /*group*/, float /*softplus threshold*/)>;
[[nodiscard]] QwenGdnDecodeGatesKernel make_qwen_gdn_decode_gates_kernel(QwenWeightFormat format) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_decode_gates_dispatch(
    uint64_t heads, uint64_t k, QwenWeightFormat format, uint32_t group = 64u);

// Q/K L2 norm + Q scale over the conv+SiLU output. One 32-lane warp per
// (token, qk_head, q_or_k) row of head_dim: x*rsqrt(sum(x*x)+eps) (FLA/HF
// l2norm, eps inside the rsqrt). q/k out [T, qk_heads, hd]; v is NOT touched
// here (the recurrent kernel reads it from the conv output at its packed
// offset). q_scale = 1/sqrt(head_dim) applied to Q only.
using QwenGdnQkNormKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*conv_out [T, 2*qk_n + v_n] packed QKV*/,
    compute::Buffer<float> /*q out [T,qk_heads,hd]*/,
    compute::Buffer<float> /*k out [T,qk_heads,hd]*/,
    uint32_t /*tokens*/, uint32_t /*qk_heads*/, uint32_t /*head_dim*/,
    uint32_t /*v_channels*/, float /*eps*/, float /*q_scale*/,
    uint32_t /*rows = tokens*qk_heads*/)>;
[[nodiscard]] QwenGdnQkNormKernel make_qwen_gdn_qk_l2norm_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_qk_norm_dispatch(uint64_t tokens, uint64_t qk_heads);

// FP32 delta-rule recurrence over T tokens (T=1 decode, T>1 recurrent
// prefill). One BLOCK (128 threads, 4 warps) per V head; warp w owns value
// columns [w*hd/4, (w+1)*hd/4). Lane l of warp w owns column v = w*hd/4 + l
// (lanes beyond hd/4 contribute nothing) and keeps the state column S[:,v]
// in a thread-local array across the whole token loop -- loaded once,
// decayed/read/updated in registers, written back at the end. k/q vectors
// are staged in shared memory per token (two sync_block per token); the
// err/output dots are PER-LANE serial reductions over the key axis (each
// lane owns a different column, so no warp collective mixes them). q/k are
// the L2-normalized QK-head tensors (tiled GGUF order: V head h uses QK head
// h % qk_heads);
// v is read from the packed conv output at v_offset + h*hd. beta/g are per
// V head. State [v_heads,hd,hd] (S[k,v]) updated in place; out [T,v_heads,hd].
// head_dim must be a multiple of 32 in [32,128] (one state column per lane).
using QwenGdnRecurrentKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q [T,qk_heads,hd] normalized+scaled*/,
    compute::Buffer<float> /*k [T,qk_heads,hd] normalized*/,
    compute::Buffer<float> /*conv_out [T, qkv_channels] (V source)*/,
    compute::Buffer<float> /*beta [T,v_heads]*/,
    compute::Buffer<float> /*g [T,v_heads]*/,
    compute::Buffer<float> /*state [v_heads,hd,hd] in/out*/,
    compute::Buffer<float> /*out [T,v_heads,hd]*/,
    uint32_t /*tokens*/, uint32_t /*v_heads*/, uint32_t /*head_dim*/,
    uint32_t /*qk_heads*/, uint32_t /*v_offset (2*qk_heads*hd)*/)>;
[[nodiscard]] QwenGdnRecurrentKernel make_qwen_gdn_recurrent_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_recurrent_dispatch(uint64_t v_heads);
// Qwen decode T=1/head_dim=128 specialization. Four blocks per V head split
// the key axis into 32-row slices (128 blocks total for 32 heads). Each thread
// owns 32 state elements; four key slices reduce with two XOR warp shuffles.
// This trades a different FP32 reduction order for 4x grid parallelism and
// 16x less per-thread local state than the original 512-entry allocation.
[[nodiscard]] QwenGdnRecurrentKernel make_qwen35_gdn_decode4_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_gdn_decode4_dispatch(uint64_t v_heads);
[[nodiscard]] Status validate_qwen35_gdn_decode4_inputs(
    uint64_t tokens, uint64_t v_heads, uint64_t qk_heads, uint64_t head_dim);
[[nodiscard]] Status validate_qwen_gdn_recurrent_inputs(uint64_t tokens, uint64_t v_heads,
                                                        uint64_t qk_heads, uint64_t head_dim);
// Shared-memory bytes required by the recurrent kernel per block
// (2*hd floats for the staged k/q vectors).
[[nodiscard]] uint64_t qwen_gdn_recurrent_shared_bytes(uint64_t head_dim) noexcept;
// Check against a backend's per-block shared-memory limit.
[[nodiscard]] Status check_qwen_gdn_shared_budget(uint64_t head_dim, uint64_t backend_shared_bytes);

// FP32 WY representation + UT transform over serial 64-token chunks. One
// 128-thread block owns one V-head and carries its K-major state across every
// chunk. The kernel computes chunk-local G=cumsum(g), the decay-scaled strict
// lower KKT matrix, its unit-lower inverse by forward substitution, W and the
// unscaled new-value matrix, then emits outputs from the chunk-start state and
// applies the gated low-rank state update. `w` and `v_new` are caller-owned
// scratch [T,v_heads,head_dim]; `out` has the same extent. Ragged tails are
// masked exactly, and q arrives normalized + scaled just like the recurrent
// path. This is algebraically equivalent to the per-token recurrence, with a
// different FP32 rounding order.
using QwenGdnChunkKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q*/, compute::Buffer<float> /*k*/,
    compute::Buffer<float> /*conv_out*/, compute::Buffer<float> /*beta*/,
    compute::Buffer<float> /*g*/, compute::Buffer<float> /*state in/out*/,
    compute::Buffer<float> /*w scratch*/, compute::Buffer<float> /*v_new scratch*/,
    compute::Buffer<float> /*out*/, uint32_t /*tokens*/, uint32_t /*v_heads*/,
    uint32_t /*head_dim*/, uint32_t /*qk_heads*/, uint32_t /*v_offset*/)>;
inline constexpr uint32_t qwen_gdn_chunk_size = 64u;
[[nodiscard]] QwenGdnChunkKernel make_qwen_gdn_chunk_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_chunk_dispatch(uint64_t v_heads);
[[nodiscard]] Status validate_qwen_gdn_chunk_inputs(uint64_t tokens, uint64_t v_heads,
                                                    uint64_t qk_heads, uint64_t head_dim);
[[nodiscard]] uint64_t qwen_gdn_chunk_shared_bytes() noexcept;
[[nodiscard]] Status dsl_ops_chunk_prefill_status() noexcept;

// Gated output RMSNorm: y = o*rsqrt(mean(o^2)+eps) * norm_w * silu(z), per
// (token, v_head) row of head_dim. DIRECT weight; norm before gate. One
// 32-lane warp per row.
using QwenGdnGatedNormKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*o [T,v_heads,hd]*/,
    compute::Buffer<float> /*z [T,v_heads*hd] (attn_gate projection)*/,
    compute::Buffer<float> /*norm_w [hd]*/,
    compute::Buffer<float> /*out [T,v_heads,hd]*/,
    uint32_t /*tokens*/, uint32_t /*heads*/, uint32_t /*head_dim*/, float /*eps*/)>;
[[nodiscard]] QwenGdnGatedNormKernel make_qwen_gdn_gated_norm_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_gated_norm_dispatch(uint64_t tokens, uint64_t heads);

// ---------------------------------------------------------------------------
// 9. Independent CPU oracles (double accumulation; NO device/DSL calls).
//    All validate sizes and never read OOB. They are deliberately separate
//    code paths from the kernels (scalar loops vs warp/block collectives).
// ---------------------------------------------------------------------------
[[nodiscard]] Status ref_qwen_embedding(QwenWeightFormat format, luisa::span<const uint32_t> ids,
                                           luisa::span<const uint32_t> table, luisa::span<const float> scales,
                                           uint64_t hidden, uint64_t vocab, uint32_t group,
                                           luisa::vector<float> &out);
  [[nodiscard]] Status ref_qwen_rmsnorm(luisa::span<const float> x, luisa::span<const float> w,
                                       luisa::span<const float> residual, bool has_residual,
                                       uint32_t rows, uint32_t dim, float eps,
                                       luisa::vector<float> &y, luisa::vector<float> &sum_out);
  // Snapshot epilogue oracle: sum_out = x bit-exactly; y = plain RMSNorm of
  // the snapshot (direct pre-added weight). Separate code path from
  // ref_qwen_rmsnorm on purpose (independent verification).
  [[nodiscard]] Status ref_qwen_rmsnorm_snapshot(luisa::span<const float> x, luisa::span<const float> w,
                                                uint32_t rows, uint32_t dim, float eps,
                                                luisa::vector<float> &y, luisa::vector<float> &sum_out);
[[nodiscard]] Status ref_qwen_matmul(QwenWeightFormat format, luisa::span<const float> x,
                                        luisa::span<const uint32_t> w, luisa::span<const float> scales,
                                        luisa::span<const float> residual, bool has_residual,
                                        uint32_t m, uint32_t n, uint32_t k, uint32_t group,
                                        luisa::vector<float> &y);
[[nodiscard]] Status ref_qwen_swiglu(luisa::span<const float> gate, luisa::span<const float> up,
                                        luisa::vector<float> &out);
[[nodiscard]] Status ref_qwen_swiglu_fused(luisa::span<const float> proj, uint32_t m, uint32_t f,
                                              luisa::vector<float> &out);
// Full attention prepare oracle: split/norm/rope (+optional append). Mirrors
// the kernel contract; caches are caller vectors sized capacity*kv_heads*hd.
[[nodiscard]] Status ref_qwen_attn_prepare(luisa::span<const float> q_proj, luisa::span<const float> k_proj,
                                              luisa::span<const float> v_proj, luisa::span<const float> q_norm_w,
                                              luisa::span<const float> k_norm_w, luisa::span<const int32_t> positions,
                                              uint32_t tokens, const QwenDslConfig &cfg, uint32_t pos0,
                                              bool append_kv, luisa::span<float> k_cache, luisa::span<float> v_cache,
                                              luisa::vector<float> &q, luisa::vector<float> &gate,
                                              luisa::vector<float> &k, luisa::vector<float> &v);
[[nodiscard]] Status ref_qwen_rope_partial(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                              uint32_t tokens, uint32_t heads, uint32_t head_dim,
                                              uint32_t rope_dim, float theta, luisa::vector<float> &out);
// Causal GQA prefill oracle (naive two-pass softmax over the cache).
[[nodiscard]] Status ref_qwen_attn_prefill(luisa::span<const float> q, luisa::span<const float> gate,
                                              luisa::span<const float> k_cache, luisa::span<const float> v_cache,
                                              uint32_t tokens, uint32_t q_heads, uint32_t kv_heads,
                                              uint32_t head_dim, float scale, uint32_t pos0, uint32_t kv_len,
                                              luisa::vector<float> &out);
// Decode oracle (T=1 specialization of the same math).
[[nodiscard]] Status ref_qwen_attn_decode(luisa::span<const float> q, luisa::span<const float> gate,
                                             luisa::span<const float> k_cache, luisa::span<const float> v_cache,
                                             uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                                             float scale, uint32_t kv_len, luisa::vector<float> &out);
[[nodiscard]] Status ref_qwen_gdn_conv(luisa::span<const float> src, luisa::span<const float> weight,
                                          luisa::span<float> ring, uint32_t tokens, uint32_t channels,
                                          uint32_t kernel_size, uint32_t pos0, luisa::vector<float> &out);
[[nodiscard]] Status ref_qwen_gdn_gates(luisa::span<const float> a, luisa::span<const float> b,
                                           luisa::span<const float> ssm_a, luisa::span<const float> dt_bias,
                                           uint32_t tokens, uint32_t heads, float softplus_threshold,
                                           luisa::vector<float> &beta, luisa::vector<float> &g);
[[nodiscard]] Status ref_qwen_gdn_qk_norm(luisa::span<const float> conv_out, uint32_t tokens,
                                             uint32_t qk_heads, uint32_t head_dim, uint32_t v_channels,
                                             float eps, float q_scale,
                                             luisa::vector<float> &q, luisa::vector<float> &k);
[[nodiscard]] Status ref_qwen_gdn_recurrent(luisa::span<const float> q, luisa::span<const float> k,
                                               luisa::span<const float> conv_out, luisa::span<const float> beta,
                                               luisa::span<const float> g, luisa::span<float> state,
                                               uint32_t tokens, uint32_t v_heads, uint32_t head_dim,
                                               uint32_t qk_heads, luisa::vector<float> &out);
[[nodiscard]] Status ref_qwen_gdn_gated_norm(luisa::span<const float> o, luisa::span<const float> z,
                                                luisa::span<const float> norm_w, uint32_t tokens,
                                                uint32_t heads, uint32_t head_dim, float eps,
                                                luisa::vector<float> &out);

// ---------------------------------------------------------------------------
// 10. Device-free self-test: deterministic nontrivial data + ragged shapes
//     (odd K/N/M, partial fp8 groups, tail words, zero groups, conv history
//     across chunk boundaries, GQA mapping, partial-RoPE tails, delta-rule
//     recurrence vs an independent scalar reference). Runs the CPU oracles
//     and host packing round-trips; no Device needed. Returns ok() when all
//     checks pass; `report` receives one line per check.
// ---------------------------------------------------------------------------
[[nodiscard]] Status dsl_ops_self_test(luisa::vector<luisa::string> *report = nullptr);

}// namespace luisa::qwen::dsl
