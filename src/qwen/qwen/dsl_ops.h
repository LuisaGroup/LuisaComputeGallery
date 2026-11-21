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
// Three explicitly selectable packed weight formats; one is NEVER silently
// substituted for another:
//   bf16       : Buffer<uint32_t> of K*N/2 words. Two BF16 code units per
//                little-endian word, LOW element first. Raw GGUF BF16 bytes.
//   fp8_e4m3   : Buffer<uint32_t> of ceil(K*N/4) words. Four E4M3 bytes per
//                little-endian word, LOW element first, plus FP32 scales:
//                one scale per contiguous group of `group` K elements per
//                output row (canonical group = 64). Dequantized weight =
//                fp8(code) * scale[row * ceil(K/group) + k/group].
//                E4M3 decoding is exact (finite codes incl. subnormals;
//                S.1111.111 is NaN per the OCP FN convention).
//   q8_0       : Buffer<uint32_t> of K*N/4 words. Four SIGNED int8 codes per
//                little-endian word, LOW element first -- a verbatim byte copy
//                of the GGUF Q8_0 block payload -- plus FP32 scales: one scale
//                per contiguous group of 32 K elements per output row (GGUF
//                block size; K/32 per row), each the exact FP16->FP32 widening
//                of the checkpoint's FP16 block scale. Dequantized weight =
//                int8(code) * scale[row * (K/32) + k/32]; the code-to-float
//                map is the integer sign extension, so no conversion table and
//                no rounding is involved. Q8_0 is NEVER re-quantized from BF16
//                and NEVER routed through FP8; `group` for this format is
//                always 32 and K % 32 == 0 is enforced (see
//                validate_qwen_q8_0_geometry).
//   w4         : Buffer<uint32_t> of K*N/8 words. EIGHT signed 4-bit
//                two's-complement codes per little-endian word, LOW nibble
//                first -- a symmetric int4 code in [-8, 7] with NO zero point --
//                plus FP32 scales: one scale per contiguous group of `group`
//                K elements per output row (canonical group 128; 64 and 32 are
//                supported where K % group == 0). Dequantized weight =
//                int4(code) * scale[row * (K/group) + k/group], where
//                int4(code) is the exact integer sign extension
//                ((code ^ 8) - 8). The scales come from a DETERMINISTIC host
//                clipping search per group (not plain round-to-nearest; see
//                host_pack_w4_matrix). W4 is only ever produced from an F32/BF16
//                source by the offline packer: it is NEVER substituted for
//                FP8/Q8_0/BF16 and never silently replaced by them.
//                K % 8 == 0 (whole words) and K % group == 0 (whole groups) are
//                enforced fail-closed.
// Matrices are stored ROW-PER-OUTPUT, K CONTIGUOUS: W[n*K + k] is the weight
// of output n, input k (GGUF [K,N] payload row stride K). y[m,n] =
// sum_k x[m,k] * W[n,k]. Activations, accumulators, norms, softmax, gates and
// GDN state are FP32 everywhere; bf16/fp8/q8_0 denotes WEIGHT STORAGE only.
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
//   * FP16/INT4 weight storage, paged KV, fused gate/up PREFILL GEMM
//     epilogue (decode fuses it; prefill SwiGLU is separate). Host sampling and
//     NextN live in engine/speculative modules.
// Q8_0 weights (WeightStorage::q8_0_packed) are implemented for every matrix
// path: embedding gather, decode GEMV (both the 4-accumulator model
// specialization and the 128-bit uint4 realization), the reference GEMM, the
// weight-reusing MT prefill GEMM, the fused FFN decode and the fused GDN
// alpha/beta gate GEMV, plus the CPU oracles.
// Split-KV decode with GQA dedup is IMPLEMENTED here (opt_plan/03 section 4.2,
// make_qwen_attn_decode_split_* + make_qwen_attn_decode_combine_*); the legacy
// per-q_head kernel stays as the short-context path and the reference.
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
#include "qwen/kv_cache_layout.h"// KV dtype/layout contract shared with the allocator

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
    uint32_t fp8_group{64};     // per-row scale group along K (fp8: 64; q8_0: 32)

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
    // Native Q8_0: 4 signed int8 codes per word + one FP32 scale per 32-element
    // K group per output row (see the storage ABI note at the top of this file).
    q8_0 = 2,
    // W4A16: 8 signed int4 codes per word + one FP32 scale per `group` K
    // elements per output row (canonical 128). See the storage ABI note above.
    w4 = 3,
};
[[nodiscard]] luisa::string_view qwen_weight_format_name(QwenWeightFormat format) noexcept;

// Packed codes per uint32 word: 2 (bf16), 4 (fp8_e4m3), 4 (q8_0), 8 (w4).
[[nodiscard]] constexpr uint32_t qwen_weight_pack(QwenWeightFormat format) noexcept {
    return format == QwenWeightFormat::bf16 ? 2u
           : format == QwenWeightFormat::w4  ? 8u
                                             : 4u;
}
// True when the format stores one scale per K group (fp8_e4m3, q8_0).
[[nodiscard]] constexpr bool qwen_weight_is_scaled(QwenWeightFormat format) noexcept {
    return format != QwenWeightFormat::bf16;
}
// Canonical scale group along K for a format: 64 for fp8_e4m3, 32 for q8_0
// (the GGUF Q8_0 block), 0 for bf16 (no scales).
inline constexpr uint32_t kQwenQ8BlockSize = 32u;// == qwen::kQ8_0BlockSize
inline constexpr uint64_t kQwenQ8BlockBytes = 34u;// 2-byte FP16 scale + 32 int8
// Canonical W4 scale group along K (validate_qwen_w4_geometry also accepts 32
// and 64 where the geometry allows them).
inline constexpr uint32_t kQwenW4DefaultGroup = 128u;
inline constexpr uint32_t kQwenW4CodesPerWord = 8u;
[[nodiscard]] constexpr uint32_t qwen_weight_group(QwenWeightFormat format) noexcept {
    return format == QwenWeightFormat::q8_0 ? kQwenQ8BlockSize
           : format == QwenWeightFormat::fp8_e4m3 ? 64u
           : format == QwenWeightFormat::w4      ? kQwenW4DefaultGroup
                                                 : 0u;
}

// Packed word counts for a K x N row-per-output matrix. Fails closed on
// overflow. bf16 requires an EVEN element count (the contract guarantees even
// K for this model); fp8 pads the last word with zero bytes; q8_0 requires
// K % 32 == 0 (whole 32-weight blocks and whole 8-word groups); w4 requires
// K % 8 == 0 (whole 8-code words).
[[nodiscard]] Result<uint64_t> packed_matrix_words(QwenWeightFormat format, uint64_t k, uint64_t n);
// FP8 scale element count: n * ceil(k/group).
[[nodiscard]] Result<uint64_t> fp8_scale_elements(uint64_t k, uint64_t n, uint32_t group);
// Q8_0 scale element count: n * (k/32); fails closed when k % 32 != 0.
[[nodiscard]] Result<uint64_t> q8_0_scale_elements(uint64_t k, uint64_t n);
// Scale element count of a format/group pair (0 for bf16, i.e. no scales).
// w4 requires a group in {32, 64, 128} and K % group == 0, so its count is
// exactly n * (K / group); q8_0 is fixed at K / 32.
[[nodiscard]] Result<uint64_t> weight_scale_elements(QwenWeightFormat format, uint64_t k,
                                                     uint64_t n, uint32_t group);
// Fail-closed geometry validation for a Q8_0 matrix: K % 32 == 0, K > 0 and a
// group that is exactly the GGUF block size are the only accepted shapes.
[[nodiscard]] Status validate_qwen_q8_0_geometry(uint64_t k, uint32_t group) noexcept;

// Fail-closed geometry validation for a W4 matrix: group in {32, 64, 128},
// K > 0, K % 8 == 0 (whole packed words) and K % group == 0 (whole groups).
// Ragged K is refused, never padded silently.
[[nodiscard]] Status validate_qwen_w4_geometry(uint64_t k, uint32_t group) noexcept;

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
  // q8_0 is REFUSED here: the format is only ever populated from a GGUF block
  // payload (host re-quantization would silently double-quantize); use
  // host_pack_q8_0_codes with already-quantized blocks instead.
// w4 uses the canonical clipping search (0.85..1.0 in 31 steps) and fills
// `scales` (n*(k/group)); host_pack_w4_matrix exposes the full config.
  [[nodiscard]] Status host_pack_matrix(QwenWeightFormat format, luisa::span<const float> w,
                                        uint64_t k, uint64_t n, uint32_t group,
                                        luisa::vector<uint32_t> &words,
                                        luisa::vector<float> &scales);
  // Exact host decode of one int4 nibble -> FP32 value in [-8, 7]
// (((nibble ^ 8) - 8), the same integer sign extension the kernels use).
[[nodiscard]] float host_w4_code_to_f32(uint32_t nibble) noexcept;
// Quantize a whole [n,k] FP32 row-per-output matrix to the W4 ABI with the
// deterministic per-group clipping search: for each group of `group`
// consecutive K elements the max-abs scale (max_abs/7) is multiplied by a grid
// of `steps` ratios in [min_ratio, max_ratio] and the ratio minimizing the
// group's dequantization MSE (computed in double against the STORED FP32
// scale) wins; ties keep the smaller ratio. Reproducible: identical input and
// config produce byte-identical words and scales. `mean_ratio`, `mse` and
// `rtn_mse` are optional diagnostics (mean chosen ratio, mean squared error of
// the search, and of plain max-abs RTN for comparison).
[[nodiscard]] Status host_pack_w4_matrix(luisa::span<const float> w,
                                         uint64_t k, uint64_t n, uint32_t group,
                                         float min_ratio, float max_ratio, uint32_t steps,
                                         luisa::vector<uint32_t> &words,
                                         luisa::vector<float> &scales,
                                         float *mean_ratio = nullptr,
                                         double *mse = nullptr,
                                         double *rtn_mse = nullptr);
// Encode ALREADY-QUANTIZED Q8_0 blocks into the device ABI (tests, offline
  // tools and the loader's shared primitive). `codes` holds the raw signed int8
  // block codes in element order and `scale_bits` the IEEE FP16 scale of each
  // 32-element block (k/32 per row, row-major). Codes are copied byte for byte
  // and each scale is widened exactly to FP32.
  [[nodiscard]] Status host_pack_q8_0_codes(luisa::span<const uint8_t> codes,
                                            luisa::span<const uint16_t> scale_bits,
                                            uint64_t k, uint64_t n,
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
    compute::Buffer<float> /*scales (fp8/q8_0; bind 1-elem dummy for bf16)*/,
    compute::Buffer<float> /*out [tokens,hidden]*/,
    uint32_t /*tokens*/, uint32_t /*hidden*/, uint32_t /*vocab*/,
    uint32_t /*group (fp8/q8_0)*/, uint32_t /*padded_cols*/)>;
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
// Register-cached model specialization (opt_plan/09 N3.1). `dim` and
// `block_size` are COMPILE-TIME, so each thread owns exactly dim/block_size
// elements (10 at the model geometry hidden = 2560, block = 256) in a
// constant-indexed register array. The sum-of-squares pass, the residual /
// snapshot store and the normalization then touch global memory ONCE: the
// generic realizations above re-read x (plain) or sum_out (residual and
// snapshot) in a second global pass after the block reduction, and re-read the
// norm weight in both passes. Same FP32 order (per-thread elements ascending,
// then the block's warp partials in warp order) and the same epilogue
// expression, so the result is numerically identical to the wide kernel of the
// same block size; `dim % block_size == 0` is required (the model's 2560
// divides 256 and 128 exactly). The norm weight is cached in registers too.
// The runtime `dim` argument is still checked against the compiled one, so a
// mismatched geometry is a guarded no-op instead of a misread.
  inline constexpr uint32_t qwen35_rmsnorm_model_dim = 2560u;
  [[nodiscard]] QwenRmsNormKernel make_qwen35_rmsnorm_reg_kernel(
      QwenNormEpilogue epilogue, uint32_t dim, uint32_t block_size) noexcept;
  [[nodiscard]] Result<QwenLaunch1D> plan_qwen35_rmsnorm_reg_dispatch(
      uint64_t rows, uint64_t dim, uint32_t block_size);

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
    compute::Buffer<float> /*scales (fp8/q8_0; dummy for bf16)*/,
    compute::Buffer<float> /*residual [M,N] (dummy when none)*/,
    compute::Buffer<float> /*y [M,N]*/,
    uint32_t /*M*/, uint32_t /*N*/, uint32_t /*K*/, uint32_t /*group (fp8/q8_0)*/)>;
[[nodiscard]] QwenGemvKernel make_qwen_gemv_kernel(QwenWeightFormat format,
                                                   QwenMatmulEpilogue epilogue) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gemv_dispatch(uint64_t m, uint64_t n, uint64_t k);
  // Qwen3.5-4B decode-only M=1 realization. Four independent packed-word
  // accumulators per lane expose memory-level parallelism; FP8 uses an exact
  // 256-entry shared LUT instead of per-weight exponent normalization loops.
  // Q8_0 decodes each int8 code with the integer sign extension
  // (((code ^ 0x80) - 128) as float), so no table and no rounding is involved,
  // and reads one FP32 block scale per 8 words (32 K elements).
  // The real K dimensions (2560/4096/9216) divide 128 packed words exactly, so
  // the inner loop has no tail predicates. The generic factory above remains
  // the ragged/M<=4 reference. Reduction order differs within FP32 tolerance.
  [[nodiscard]] QwenGemvKernel make_qwen35_decode_gemv_kernel(
      QwenWeightFormat format, QwenMatmulEpilogue epilogue) noexcept;
      [[nodiscard]] Result<QwenLaunch1D> plan_qwen35_decode_gemv_dispatch(
        uint64_t n, uint64_t k, QwenWeightFormat format, uint32_t group = 0u);

  // 3a-bis. MERGED decode input projections (opt_plan/03 section 3 + D5).
  //
  // One warp per OUTPUT ROW of an offline-merged matrix (qwen/projection_pack.h),
  // with segment-aware destinations, so an attention or GDN layer pays ONE launch
  // for its whole input-projection group instead of one per projection. The rows of
  // the merged matrix are the source rows in the canonical segment order, and every
  // row is computed by the SAME dot realization the unmerged path uses for it:
  // * projection rows (QKV/Z or Q/K/V) use the decode projection realization of the
  //   format exactly as make_qwen35_decode_gemv_kernel / make_qwen_gemv_kernel do
  //   (BF16: the generic packed pair loop; FP8/Q8_0: the four-stream specialization
  //   with the shared E4M3 LUT), INCLUDING the ragged-K fallback branch, so the
  //   per-output FP32 accumulation order is unchanged and the logits stay
  //   bit-identical to the unmerged session;
  // * the GDN alpha/beta rows use the fused gate kernel's own dot and epilogue
  //   (make_qwen_gdn_decode_gates_kernel), i.e. g = ssm_a * softplus(alpha + dt_bias)
  //   and beta = sigmoid(beta), so the merged launch REPLACES that launch instead of
  //   adding a pass over a 12352-wide intermediate.
  // No intermediate buffer is ever written: each warp stores its scalar directly
  // into the row's real destination.
  using QwenMergedGdnGemvKernel = compute::Kernel1D<void(
      compute::Buffer<float> /*x [K] (M == 1)*/,
      compute::Buffer<uint32_t> /*W merged [N, K] row-per-output*/,
      compute::Buffer<float> /*scales (fp8/q8_0; dummy for bf16)*/,
      compute::Buffer<float> /*qkv_out [qkv_rows]*/,
      compute::Buffer<float> /*z_out [z_rows]*/,
      compute::Buffer<float> /*ssm_a [gate_heads]*/,
      compute::Buffer<float> /*dt_bias [gate_heads]*/,
      compute::Buffer<float> /*beta_out [gate_heads] (dummy allowed)*/,
      compute::Buffer<float> /*g_out [gate_heads] (dummy allowed)*/,
      uint32_t /*K*/, uint32_t /*group (fp8/q8_0)*/,
      uint32_t /*qkv_rows*/, uint32_t /*z_rows*/, uint32_t /*gate_heads*/,
      float /*softplus_threshold*/)>;
  // The projection-row dot realization follows `format` (BF16: the generic packed
  // pair loop the engine compiles for BF16 decode; FP8/Q8_0: the four-stream
  // specialization), i.e. the SAME realization the unmerged decode route would use
  // for the same row, so the two agree bit for bit. The alpha/beta rows always use
  // the fused gate realization, which is format-independent.
  [[nodiscard]] QwenMergedGdnGemvKernel make_qwen35_merged_gdn_gemv_kernel(
      QwenWeightFormat format) noexcept;
  // Geometry contract (fail closed, host only): K must satisfy the decode GEMV's
  // packed contract for `format`, every segment must be non-empty, and the merged
  // row count must fit the uint32/launch envelope. Returns the launch shape:
  // ceil(N / 4) blocks of 128 threads (4 warps, one output row per warp).
  [[nodiscard]] Result<QwenLaunch1D> plan_qwen35_merged_gdn_gemv_dispatch(
      uint64_t qkv_rows, uint64_t z_rows, uint64_t gate_heads, uint64_t k,
      QwenWeightFormat format, uint32_t group = 0u);

  using QwenMergedAttnGemvKernel = compute::Kernel1D<void(
      compute::Buffer<float> /*x [K] (M == 1)*/,
      compute::Buffer<uint32_t> /*W merged [N, K] row-per-output*/,
      compute::Buffer<float> /*scales (fp8/q8_0; dummy for bf16)*/,
      compute::Buffer<float> /*q_out [q_rows]*/,
      compute::Buffer<float> /*k_out [kv_rows]*/,
      compute::Buffer<float> /*v_out [kv_rows]*/,
      uint32_t /*K*/, uint32_t /*group (fp8/q8_0)*/,
      uint32_t /*q_rows*/, uint32_t /*kv_rows*/)>;
  [[nodiscard]] QwenMergedAttnGemvKernel make_qwen35_merged_attn_gemv_kernel(
      QwenWeightFormat format) noexcept;
  [[nodiscard]] Result<QwenLaunch1D> plan_qwen35_merged_attn_gemv_dispatch(
      uint64_t q_rows, uint64_t kv_rows, uint64_t k,
      QwenWeightFormat format, uint32_t group = 0u);

// 3b. VECTORIZED 128-bit decode GEMV (opt_plan/09 N2).
// Same math/ABI contract as make_qwen35_decode_gemv_kernel (M == 1, one warp per
// output row, warp_active_sum, optional residual), but the WEIGHTS are bound as
// uint4 and the ACTIVATION as float4:
//   * one LDG.E.128 covers 16 FP8 (or 8 BF16) weights instead of 4 (2), i.e. 4x
//     (2x) fewer weight load instructions for the same bytes;
//   * the activation is read with float4 loads (4x fewer instructions, and the
//     per-lane 16-float window no longer costs 16 scalar sector touches);
  // * FP8 scale: a uint4 of 16 weights provably sits inside one 64-element
  // quantisation group, so a group index is `vec / (group/16)` -- one scale
  // read per 16 weights instead of one per 4;
  // * Q8_0: the same uint4 geometry (16 int8 codes = 16 weights), one FP32
  // block scale per 2 uint4s (32 K elements), and the same branchless
  // sign-extension decode -- no LUT, no shared memory, no barrier;
  // * FP8 E4M3 -> FP32 is the BRANCHLESS, LUT-FREE decode below, bit-identical
  // to qwen_fp8_e4m3_to_f32 for every one of the 256 codes (subnormals,
  // signed zero and the S.1111.111 -> NaN policy included), so the 256-entry
  // shared LUT and its block barrier disappear.
  // The generic 32-bit realization above stays compiled as the reference path and
  // as the ragged/tail fallback: this family requires K to divide 32 vec4s
  // (FP8/Q8_0: K % 512 == 0, BF16: K % 256 == 0) and, for the scaled formats,
  // group % 16 == 0 with group | K. Anything else is a planning FAILURE, never a
  // silent fallback.
  // The accumulation PARTITION changes (a lane owns 4 consecutive packed words
  // per step instead of four words 32 apart), so results are NOT bit-identical to
  // the 32-bit decode kernel -- same products, different FP32 summation order.
  using QwenGemvPackedKernel = compute::Kernel1D<void(
      compute::Buffer<float4> /*x [K] as float4*/,
      compute::Buffer<uint4> /*W packed as uint4 (4 words)*/,
      compute::Buffer<float> /*scales (fp8/q8_0; dummy for bf16)*/,
      compute::Buffer<float> /*residual [N] or dummy*/,
      compute::Buffer<float> /*y [N]*/,
      uint32_t /*N*/, uint32_t /*K*/, uint32_t /*group (fp8/q8_0)*/,
      uint32_t /*vecs_per_row = K / elements_per_uint4*/)>;
  [[nodiscard]] QwenGemvPackedKernel make_qwen35_decode_gemv_packed_kernel(
      QwenWeightFormat format, QwenMatmulEpilogue epilogue) noexcept;
  [[nodiscard]] Result<QwenLaunch1D> plan_qwen35_decode_gemv_packed_dispatch(
      uint64_t n, uint64_t k, QwenWeightFormat format, uint32_t group);
  // Elements covered by one uint4 for a weight format (16 FP8, 16 Q8_0, 8 BF16).
    [[nodiscard]] constexpr uint32_t qwen_packed_vec_elements(QwenWeightFormat format) noexcept {
      // Elements covered by one uint4: 8 BF16, 16 FP8/Q8_0 codes, 32 W4 codes
      // (W4 has NO 128-bit realization -- see validate_packed_decode_geometry).
      return format == QwenWeightFormat::bf16 ? 8u
             : format == QwenWeightFormat::w4  ? 32u
                                               : 16u;
  }

// ---------------------------------------------------------------------------
// 4. Packed GEMM prefill. Same math/ABI as GEMV; one thread per (m,n) output,
//    serial FP32 dot over K (weights of one output row are contiguous).
//    Correctness-first baseline; a tiled/shared-memory variant is future work.
// ---------------------------------------------------------------------------
using QwenGemmKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*x [M,K]*/,
    compute::Buffer<uint32_t> /*W packed [N,K]*/,
    compute::Buffer<float> /*scales (fp8/q8_0; dummy for bf16)*/,
    compute::Buffer<float> /*residual [M,N] (dummy when none)*/,
    compute::Buffer<float> /*y [M,N]*/,
    uint32_t /*M*/, uint32_t /*N*/, uint32_t /*K*/, uint32_t /*group (fp8/q8_0)*/,
    uint32_t /*padded_cols*/)>;
[[nodiscard]] QwenGemmKernel make_qwen_gemm_kernel(QwenWeightFormat format,
                                                   QwenMatmulEpilogue epilogue) noexcept;
[[nodiscard]] Result<QwenLaunch2D> plan_qwen_gemm_dispatch(uint64_t m, uint64_t n, uint64_t k);

// ---------------------------------------------------------------------------
// 4b. Model-path WEIGHT-REUSING prefill GEMM (opt_plan/02 section 3, M-b).
//
// Defect removed: make_qwen_gemm_kernel above is one thread per (m, n) output
// with a serial K loop, so (a) adjacent lanes own adjacent output columns and
// therefore read weight rows K*pack bytes apart (one word per 32 B sector, 16
// sectors per warp instruction) and (b) the whole weight slab is re-streamed
// once per block-row group of 16 rows (4x at prefill_chunk 64, 16x at 256).
//
// Here one 32-lane WARP owns one output column n and splits K across its lanes,
// and one BLOCK owns `warps` consecutive columns and an MT-row stripe of m:
//
//   for each packed weight word (lane-stride 32 words -> fully coalesced,
//   one 128 B transaction per warp instruction):
//       decode the word ONCE (2 bf16 halves / 4 fp8 bytes, one fp8 scale)
//       for r in [0, MT): acc[r] += x[row0 + r, k] * w   (MT FMAs)
//
// so weight traffic is weight_bytes * ceil(m / MT) instead of * ceil(m / 16)
// with 8x sector amplification, and every weight word is reused MT times from
// registers. All array extents/indices are compile-time constants (MT, pack
// width), so the accumulators are register-allocated: no Local<> runtime index,
// no local-memory spill. Accumulation is FP32 and the epilogue/ABI is IDENTICAL
// to make_qwen_gemm_kernel (same ten arguments, same [M,N] output layout, same
// none/residual epilogues), so the executor only switches the factory.
//
// Shape contract (validated by the host planner, fail-closed; the reference
// GEMM above stays the path for anything else):
  // * bf16 requires K % 2 == 0 and fp8/q8_0 require K % 4 == 0, i.e. every
  // weight row starts on a packed-word boundary (same eligibility rule as the
  // GEMV packed fast path). The model K values 2560/4096/9216 all satisfy it.
  // * fp8/q8_0 additionally require group % 4 == 0 and group > 0, so one packed
  // word never straddles a scale group (one scale load per word). For q8_0 the
  // group is the GGUF block (32), i.e. one scale per 8 words.
//   * M, N are arbitrary (ragged): the last m stripe clamps its rows and the
//     last column block guards `col < n`.
// Reduction order differs from the reference GEMM (K is split across lanes and
// combined by one warp reduction), so results agree to FP32 reassociation, not
// bit-for-bit. `mt` is the only tuning knob; 16 is the measured default
// (opt_plan/02 section 3.1 register budget).
// ---------------------------------------------------------------------------
struct QwenGemmMtConfig {
    // Rows of m handled per warp tile (compile-time). Bounds the live
    // accumulator set; MT=4 covers decode/MTP verify sizes, MT=16..32 prefill.
    uint32_t mt{16};
    // Output columns per 32*warps-thread block (one warp per column).
    uint32_t warps{4};
};
[[nodiscard]] Status validate_qwen35_gemm_mt_config(const QwenGemmMtConfig &cfg) noexcept;
// Pins warp size 32; the host must check device.compute_warp_size() == 32.
[[nodiscard]] Status select_qwen35_gemm_mt_config(uint32_t warp_size, QwenGemmMtConfig &cfg) noexcept;
// Kernel2D with block (32*warps, 1) and grid (column_blocks, ceil(M/mt)), i.e.
// exactly the same ten arguments and the same 2D dispatch shape as
// make_qwen_gemm_kernel. A 1D flattened launch was measured to exceed the
// portable 65535-groups-per-dimension envelope at M=1024, N=9216 (DX rejects
// it outright), so both families use the 2D grid.
using QwenGemmMtKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*x [M,K]*/,
    compute::Buffer<uint32_t> /*W packed [N,K] row-per-output*/,
    compute::Buffer<float> /*scales (fp8/q8_0; dummy for bf16)*/,
    compute::Buffer<float> /*residual [M,N] (dummy when none)*/,
    compute::Buffer<float> /*y [M,N]*/,
    uint32_t /*M*/, uint32_t /*N*/, uint32_t /*K*/, uint32_t /*group (fp8/q8_0)*/,
    uint32_t /*column_blocks = ceil(N / warps)*/)>;
[[nodiscard]] QwenGemmMtKernel make_qwen35_gemm_mt_kernel(
    const QwenGemmMtConfig &cfg, QwenWeightFormat format,
    QwenMatmulEpilogue epilogue) noexcept;
// Host dispatch plan. One `dispatch(total_x, total_y)` issues the whole tile
// grid; `column_blocks` must be passed to the kernel as its tenth argument.
struct QwenGemmMtPlan {
    uint32_t total_x{0};// column_blocks * 32 * warps threads
    uint32_t total_y{0};// stripes thread rows
    uint32_t mt{0};
    uint32_t columns_per_block{0};
    uint32_t column_blocks{0};// ceil(N / columns_per_block)
    uint32_t stripes{0};      // ceil(M / mt)
    uint64_t blocks{0};       // column_blocks * stripes
    [[nodiscard]] bool empty() const noexcept { return blocks == 0u; }
};
[[nodiscard]] Result<QwenGemmMtPlan> plan_qwen35_gemm_mt_dispatch(
    uint64_t m, uint64_t n, uint64_t k, QwenWeightFormat format, uint32_t group,
    const QwenGemmMtConfig &cfg);

// ---------------------------------------------------------------------------
// 4c. Activation workspace geometry / aliasing plan (opt_plan/09 V6 + N3.5).
//
// The executor used to allocate ~27 separate [chunk, width] activation buffers
// simultaneously (~116k floats per chunk row, ~454 MiB at chunk 1024). Most of
// them are never live at the same time, so they are grouped into a small number
// of ARENAS: one device buffer per arena, and every member is a handle copy of
// its arena (identical allocation, so backend barriers stay correct and every
// buffer handle the executor holds stays stable).
//
// Aliasing is allowed ONLY between members whose command ranges cannot overlap.
// `qwen_workspace_members` publishes the plan and the recorded order it is
// derived from, so a host test can re-verify disjointness instead of trusting a
// comment; `qwen_workspace_geometry` is the single source of truth for the sizes
// the executor allocates.
//
// Recorded order per layer (the executor's fixed schedule):
//   full-attention layer            GDN layer                     every layer
//   1 input norm snapshot           (same)                        12 post-attn norm
//   2 q proj -> _attn_proj_q        2 packed QKV  -> _proj_a      13 ffn gate proj
//   3 k proj -> _attn_proj_k        3 z           -> _proj_b     14 ffn up proj
//   4 v proj -> _attn_proj_v        4 alpha       -> _gdn_alpha  15 swiglu
//   5 prepare -> q/gate/k/v         5 beta        -> _proj_c     16 ffn down +residual
//   6 attn   -> _attn_out           6 conv        -> _gdn_conv
//   7 out proj + residual           7 gates       -> beta/g
//                                   8 qk norm     -> _gdn_q/k
//                                   9 WY/UT chunk -> w/vnew/out
//                                  10 gated norm  -> _gdn_normed
//                                  11 out proj + residual
// Decode (rows == 1) fuses steps 4+7 (one gates kernel writes _gdn_beta/_gdn_g
// before the conv), which is why _gdn_g is NOT placed with _proj_a even though
// the prefill order alone would allow it.
// ---------------------------------------------------------------------------
inline constexpr uint32_t kQwenWorkspaceArenaCount = 10u;
enum class QwenWorkspaceStage : uint32_t {
    permanent = 0,// never aliased
    ffn = 1,      // used by the FFN block of every layer
    gdn = 2,      // used only by a GDN layer
    attention = 3,// used only by a full-attention layer
};
[[nodiscard]] luisa::string_view qwen_workspace_stage_name(QwenWorkspaceStage stage) noexcept;
struct QwenWorkspaceMember {
    const char *buffer{nullptr};  // executor member name
    const char *arena{nullptr};   // arena name ("permanent" for unaliased)
    uint32_t arena_index{0};      // >= kQwenWorkspaceArenaCount => permanent
    QwenWorkspaceStage stage{QwenWorkspaceStage::permanent};
    uint32_t first_step{0};// inclusive command step within a layer (0 = n/a)
    uint32_t last_step{0};
    uint64_t floats_per_row{0};// elements this member needs per chunk row
};
struct QwenWorkspaceGeometry {
    uint64_t rows{0};
    // Permanent, never-aliased [rows, hidden] buffers: the residual stream is
    // live across a whole layer and both residual projections read it.
    uint64_t hidden_elems{0};
    uint64_t normed_elems{0};
    uint64_t residual_elems{0};
    // Aliased pool: arena i is one device buffer of rows * arena_floats[i].
    uint64_t arena_floats[kQwenWorkspaceArenaCount]{};
    uint64_t pool_floats_per_row{0};
    uint64_t pool_elems{0};
    // Chunk-scaled allocation (permanent + pool). LM head/logits, ids, staged
    // token ids and the greedy grid buffers are fixed-size and NOT included;
    // QwenEngineInfo::workspace_bytes reports the true total.
    uint64_t chunk_elems{0};
    uint64_t chunk_bytes{0};
};
[[nodiscard]] QwenWorkspaceGeometry qwen_workspace_geometry(const QwenDslConfig &cfg,
                                                            uint64_t rows) noexcept;
// The full member -> arena/lifetime table for `cfg` (tests + logs).
[[nodiscard]] luisa::vector<QwenWorkspaceMember> qwen_workspace_members(const QwenDslConfig &cfg);

// ---------------------------------------------------------------------------
// 4d. Adaptive outer prefill chunking (opt_plan/02 section 4).
//
// chunk = clamp(next_pow2(rows / 4), 64, min(prefill_chunk_max, backend limit)),
// then min(chunk, rows) so a short prompt is one chunk. `chunks` is
// ceil(rows / chunk): the last chunk may be ragged (the executor already handles
// ragged tails and per-chunk absolute position spans, GDN WY/UT arithmetic is
// unchanged because it always processes 64-token sub-chunks internally).
// ---------------------------------------------------------------------------
[[nodiscard]] uint32_t qwen_next_pow2_u32(uint64_t v) noexcept;
// Conservative per-backend chunk ceiling. Measured, not guessed: see the phase 3
// record. Unknown backends get the smallest (DX) limit rather than the largest.
[[nodiscard]] uint32_t qwen_prefill_chunk_backend_limit(luisa::string_view backend) noexcept;
struct QwenPrefillChunkPlan {
    uint32_t chunk{0};   // rows per chunk actually selected
    uint32_t chunks{0};  // ceil(rows / chunk); the last one may be ragged
    uint32_t max_chunk{0};// effective ceiling = min(requested max, backend limit)
    uint32_t requested_max{0};
    uint32_t backend_limit{0};
    bool limited_by_backend{false};
};
[[nodiscard]] QwenPrefillChunkPlan qwen_plan_prefill_chunks(uint64_t rows, uint32_t requested_max,
                                                            uint32_t backend_limit) noexcept;

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

// 128-bit twin of the fused FFN decode kernel (opt_plan/09 N2): uint4 gate/up
// weight loads, one scale per uint4, float4 activation loads and the branchless
// LUT-free E4M3 decode (see make_qwen35_decode_gemv_packed_kernel for the exact
// contract and the accumulation-order note). Same two-accumulator-per-stream
// warp reduction and the same SiLU(gate)*up epilogue; the 32-bit kernel stays
// the reference and the ragged fallback.
using QwenFfnDecodePackedKernel = compute::Kernel1D<void(
    compute::Buffer<float4> /*x [K] as float4*/,
    compute::Buffer<uint4> /*gate W [F,K] packed*/,
    compute::Buffer<float> /*gate scales*/,
    compute::Buffer<uint4> /*up W [F,K] packed*/,
    compute::Buffer<float> /*up scales*/,
    compute::Buffer<float> /*out [F]*/,
    uint32_t /*F*/, uint32_t /*K*/, uint32_t /*group*/,
    uint32_t /*vecs_per_row*/)>;
[[nodiscard]] QwenFfnDecodePackedKernel make_qwen_ffn_decode_packed_kernel(
    QwenWeightFormat format) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_ffn_decode_packed_dispatch(
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

  // Activation SECOND-MOMENT accumulator for the W4 importance calibration
  // (qwen/w4_importance.h, opt_plan/05 section 3 item 2). One thread owns one
  // input channel j and adds the squared activation of every row of the current
  // projection input to it:
  //     acc[j] += sum over r in [0, rows) of x[r * k + j]^2
  // So the statistic is the diagonal of the empirical input Gram matrix
  // (diag(H) under independent channels): the quantity the offline packer weights
  // its group error with. The update is RACE FREE (a channel is written by exactly
  // one thread, no atomics) and DETERMINISTIC: the per-channel sum runs in row
  // order, and repeated launches add in launch order -- the same inputs always
  // produce the same bytes, which is what makes a calibration reproducible.
  // FP32 accumulation is deliberate: the statistic is a mean of bounded
  // activations over a few thousand tokens, where FP32 relative error is ~1e-5 --
  // far below the difference the weighting has to resolve.
  using QwenImportanceAccumulateKernel = compute::Kernel1D<void(
      compute::Buffer<float> /*x [rows, k]*/, compute::Buffer<float> /*acc [k]*/,
      uint32_t /*rows*/, uint32_t /*k*/)>;
  [[nodiscard]] QwenImportanceAccumulateKernel make_qwen_importance_accumulate_kernel() noexcept;
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

// ---------------------------------------------------------------------------
// Device-resident decode state (opt_plan/03 section 2.3, milestone M-c).
//
// The reference decode schedule takes the per-token scalars of a step -- the
// absolute position, the KV length the attention may read, and the greedy
// history slot -- as HOST kernel arguments (`pos0`, `kv_len`, `slot`), so the
// command list of every token differs and a CUDA graph built from it must be
// rebuilt or whole-graph updated per token. The `*_dev` kernels below read the
// same three quantities from ONE small device buffer instead. Nothing else
// changes: every other decode argument is already a device buffer or a
// compile-time-stable scalar, so with these four kernels a single recorded
// decode body stays valid for every later token (zero graph updates).
//
// Layout (qwen_dec_state_count uint32 words):
//   [position]      absolute position the current decode step consumes; the
//                   attention prepare appends its K/V there and the GDN conv
//                   kernel indexes its ring with it.
//   [kv_len]        cache entries the decode attention may read. The prepare
//                   kernel of the SAME step appended [position], so the
//                   invariant is kv_len == position + 1; the finish kernel
//                   re-derives it from the advanced position (single source of
//                   truth, no independent counter to drift).
// [history_slot]  greedy history slot the finish kernel stores the selected
//                 token in; advances by one per step.
// [eos]           0 = the last executed step selected an ordinary token that did
//                 not end generation; 1 = it selected the EOS token, the
//                 NaN/invalid sentinel (vocab) or an id >= vocab. It flags the
//                 SELECTION, not the loop, so a chunk that stopped at the context
//                 limit still reports eos == 0. Written by
//                 make_qwen_greedy_argmax_finish_while_kernel only (the per-token
//                 device-state finish kernel leaves it untouched, so a session
//                 that never runs the conditional WHILE keeps the 0 the engine
//                 uploaded).
// [remaining]     conditional-WHILE token budget of the CURRENT graph launch:
//                 the engine uploads the chunk size before each launch and the
//                 WHILE finish kernel decrements it once per executed step, so
//                 `chunk - remaining` is exactly the number of tokens the launch
//                 executed and a non-zero remainder after the launch proves the
//                 loop exited early (see the semantics note below).
// Host/device contract: the HOST remains authoritative for `position`
// (QwenEngine::info().position). The engine uploads {position, kv_len, slot}
// whenever a new session starts (reset / prefill / first capture) or when the
// caller's slot departs from the mirror, and the in-graph finish kernel then
// advances all three.
//
// CONDITIONAL WHILE semantics (opt_plan/09 N4.2). A decode chunk is ONE graph
// launch whose body is the argument-invariant device-state step and whose WHILE
// condition the body's finish kernel re-arms. Because the condition is assigned
// its launch default (1) by the engine, the body always runs at least once; the
// finish kernel then writes
//   continue = (remaining - 1) != 0 && token != eos_token && token < vocab
//              && next_position < history_capacity
// so ONE launch executes 1..chunk tokens and stops on the device: on EOS, on a
// NaN/invalid selection, at the context limit, or when the budget is spent. The
// loop can therefore never run a token past EOS and never past max_context, and
// the host learns the real trip count from [remaining] instead of assuming the
// requested chunk ran (an early exit must not look like unexecuted tokens ran).
// ---------------------------------------------------------------------------
// Word indices. Plain uint32_t constants (not an enum) because they are also
// used as kernel scalar ARGUMENTS and as buffer indices, where an enumeration
// would need a cast.
inline constexpr uint32_t qwen_dec_state_position = 0u;
inline constexpr uint32_t qwen_dec_state_kv_len = 1u;
inline constexpr uint32_t qwen_dec_state_history_slot = 2u;
inline constexpr uint32_t qwen_dec_state_eos = 3u;
inline constexpr uint32_t qwen_dec_state_remaining = 4u;
inline constexpr uint32_t qwen_dec_state_count = 5u;
// Value upload_dev_state() writes into [remaining] when the conditional WHILE is
// not in use: the per-token kernels never read or write that word, so a single
// value documents "budget not set" without a second code path.
inline constexpr uint32_t qwen_dec_state_remaining_unset = 0u;

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
// Device-state realization of the finish kernel: the history slot comes from
// state[qwen_dec_state_history_slot] instead of a host argument, and the very
// same thread that stores the token advances the state (position + 1,
// kv_len = position + 1, history_slot + 1). Selection, NaN sentinel and the
// history layout are identical to make_qwen_greedy_argmax_finish_kernel, so the
// two agree token for token. `history_capacity` bounds the slot store: a slot
// outside it is not written (the host validates the range before dispatching).
using QwenGreedyArgmaxFinishDevKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*values [256]*/, compute::Buffer<uint32_t> /*indices [256]*/,
    compute::Buffer<uint32_t> /*invalid [256]*/, compute::Buffer<uint32_t> /*next_ids [1]*/,
    compute::Buffer<uint32_t> /*history [capacity]*/, compute::Buffer<uint32_t> /*state [4]*/,
    uint32_t /*vocab*/, uint32_t /*history_capacity*/)>;
[[nodiscard]] QwenGreedyArgmaxFinishDevKernel make_qwen_greedy_argmax_finish_dev_kernel() noexcept;
// CONDITIONAL-WHILE realization of the same finish kernel (opt_plan/09 N4.2). It
// is byte-for-byte the same selection and the same state advance, plus:
//  1. [remaining] is decremented (saturating, and only by the owning thread),
//  2. [eos] records whether the selected token was the EOS token / the invalid
//     sentinel,
//  3. the enclosing WHILE condition is re-armed via set_graph_conditional() with
//     the device-computed continuation flag, so the trip count of one graph
//     launch is decided on the device.
// `cond_handle` is the cudaGraphConditionalHandle value the host created for the
// parent graph, baked into the body at build time (session constant, so the body
// stays argument-invariant). `eos_token` is the model's end-of-sequence id; pass
// a value >= vocab to disable EOS termination (the budget and the context limit
// still stop the loop). The kernel requires the CUDA device runtime (cudadevrt),
// which is why it is only ever compiled/instantiated for the WHILE graph.
using QwenGreedyArgmaxFinishWhileKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*values [256]*/, compute::Buffer<uint32_t> /*indices [256]*/,
    compute::Buffer<uint32_t> /*invalid [256]*/, compute::Buffer<uint32_t> /*next_ids [1]*/,
    compute::Buffer<uint32_t> /*history [capacity]*/, compute::Buffer<uint32_t> /*state [5]*/,
    uint32_t /*vocab*/, uint32_t /*history_capacity*/,
    uint64_t /*cond_handle*/, uint32_t /*eos_token*/)>;
[[nodiscard]] QwenGreedyArgmaxFinishWhileKernel make_qwen_greedy_argmax_finish_while_kernel() noexcept;

  // ---------------------------------------------------------------------------
  // 7. Full attention.
  //
// KV CACHE STORAGE (opt_plan/05 Q3). Every attention kernel addresses the K/V
// cache through `QwenKvStorage` below, which is the compile-time half of the
// contract in qwen/kv_cache_layout.h (the allocator owns the host half):
//   dtype  f32       -> one FP32 code per 32-bit word (the legacy reference).
//          bf16      -> two RNE BF16 codes per word, LOW code first. The kernels
//                       decode with qwen_bf16_to_f32, i.e. widen the 16-bit code,
//                       and NEVER reinterpret a packed word as a float VALUE.
//          fp8_e4m3  -> four E4M3FN codes per word, LOW code first, SCALED: the
//                       stored value is qwen_fp8_e4m3_to_f32(code) * scale, with
//                       one FP32 scale per (kv_head, 64-position block) bound as
//                       the separate k_scale/v_scale argument every attention
//                       kernel now takes (a dummy 1-element buffer for f32/bf16).
//                       The scale is written by make_qwen_kv_append_fp8_kernel
//                       (see below), never by the attention kernels.
//          All activations, accumulators and the softmax stay FP32.
//   layout position_major -> [position][kv_head][channel] (legacy/A-B).
//          blocked64      -> [kv_head][position_block=64][position%64][channel].
// The default `QwenKvStorage{}` is the legacy FP32 pair, so every existing call
// site keeps compiling against the reference path; the engine always passes the
// allocator's own dtype/layout and reports them in QwenEngineInfo.
//
// The packed store side cannot be done one code at a time: adjacent channels of
// one position share a word and the prepare kernels give adjacent channels to
// DIFFERENT lanes, so a per-code store would be a read-modify-write race. BF16
// therefore appends through the separate make_qwen_kv_append_packed_kernel() (one
// thread per WORD, both codes in the same thread) and FP8 through
// make_qwen_kv_append_fp8_kernel() (one CTA per (64-position block, kv_head),
// which also owns the block scale); the prepare kernels are compiled with
// append_kv = false on both paths. A packed word must never be addressed through
// the FP32 element arithmetic.
struct QwenKvStorage {
    QwenKvDType dtype{QwenKvDType::f32};
    QwenKvLayout layout{QwenKvLayout::position_major};
    [[nodiscard]] constexpr bool packed_words() const noexcept {
        return dtype != QwenKvDType::f32;
    }
    [[nodiscard]] constexpr bool blocked() const noexcept {
        return layout == QwenKvLayout::blocked64;
    }
    [[nodiscard]] constexpr uint32_t elements_per_word() const noexcept {
        return qwen_kv_elements_per_word(dtype);
    }
    [[nodiscard]] constexpr bool legacy() const noexcept {
        return dtype == QwenKvDType::f32 && layout == QwenKvLayout::position_major;
    }
    // True for fp8_e4m3: the region is SCALED (one FP32 scale per (kv_head,
    // 64-position block), a dense array bound separately from the sparse cache).
    [[nodiscard]] constexpr bool scaled() const noexcept { return qwen_kv_scaled(dtype); }
};
inline constexpr QwenKvStorage qwen_kv_storage_f32{};
[[nodiscard]] constexpr QwenKvStorage qwen_kv_storage_native(QwenKvDType dtype) noexcept {
    return QwenKvStorage{dtype, qwen_kv_native_layout(dtype)};
}
// Words (32-bit) of one K or V region the storage needs, block padding included;
// 0/overflow on an unusable geometry. Host mirror of the allocator's
// region_words(), used by the engine and the tests to size workspaces.
[[nodiscard]] Result<uint64_t> qwen_kv_region_words(QwenKvStorage kv, uint32_t max_context,
                                                    uint32_t kv_heads, uint32_t head_dim) noexcept;
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
[[nodiscard]] QwenAttnPrepareKernel make_qwen_attn_prepare_kernel(
    bool append_kv, bool implicit_position_ramp = false,
    QwenKvStorage kv = QwenKvStorage{}) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_prepare_dispatch(uint64_t tokens, const QwenDslConfig &cfg);
[[nodiscard]] Status validate_qwen_attn_prepare_inputs(uint64_t tokens, const QwenDslConfig &cfg,
                                                       uint64_t pos0, uint64_t cache_capacity, bool append_kv);

// Device-state realization of the same op (opt_plan/03 section 2.3): the
// absolute position comes from state[qwen_dec_state_position] and the K/V are
// ALWAYS appended there (a decode step always writes its own key/value). The
// `positions` buffer is still bound but never read, so the caller can pass the
// id staging view it already binds. Element-for-element identical to
// make_qwen_attn_prepare_kernel(true, /*implicit_position_ramp=*/true) at the
// same position.
using QwenAttnPrepareDevKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q_proj [T, 2*q_heads*hd]*/,
    compute::Buffer<float> /*k_proj [T, kv_heads*hd]*/,
    compute::Buffer<float> /*v_proj [T, kv_heads*hd]*/,
    compute::Buffer<float> /*q_norm_w [hd]*/,
    compute::Buffer<float> /*k_norm_w [hd]*/,
    compute::Buffer<int32_t> /*positions [T] (unused, bind the ids view)*/,
    compute::Buffer<float> /*q out [T, q_heads, hd]*/,
    compute::Buffer<float> /*gate out [T, q_heads, hd]*/,
    compute::Buffer<float> /*k out [T, kv_heads, hd]*/,
    compute::Buffer<float> /*v out [T, kv_heads, hd]*/,
    compute::Buffer<float> /*k_cache*/,
    compute::Buffer<float> /*v_cache*/,
    compute::Buffer<uint32_t> /*state [4]*/,
    uint32_t /*tokens*/, uint32_t /*q_heads*/, uint32_t /*kv_heads*/,
    uint32_t /*head_dim*/, uint32_t /*rope_dim*/, float /*theta*/, float /*eps*/,
    uint32_t /*cache_capacity*/,
    uint32_t /*q_rows = tokens*q_heads*/, uint32_t /*k_rows = tokens*kv_heads*/,
    uint32_t /*v_rows = tokens*kv_heads*/)>;
[[nodiscard]] QwenAttnPrepareDevKernel make_qwen_attn_prepare_dev_kernel(
    bool append_kv = true, QwenKvStorage kv = QwenKvStorage{}) noexcept;
// Model-specialized decode attention prepare (opt_plan/09 N3.2).
// The generic kernels above re-read q_proj/k_proj/q_norm_w/k_norm_w three times
// per element and evaluate the RoPE pair (exp2/log2/cos/sin) TWICE per lane,
// because a lane's two pair elements arrive as two separate strided-loop
// iterations. With head_dim 256, rope_dim 64, 16 q heads and 4 kv heads as
// COMPILE-TIME constants (the Qwen3.5-4B geometry) each lane owns exactly eight
// constant-indexed elements -- and the RoPE prefix is pairs (lane, lane+32) --
// so the norm pass caches the eight values AND the eight norm weights in
// registers, a single qwen_rope_pair per lane covers the whole rotated prefix,
// and the epilogue re-reads nothing. Layout, element order, reduction order
// (ascending per-lane j then warp_active_sum), the 1/sqrt(mean(x^2)+eps) form,
// the subsequent per-element weight multiply, the gate copy, the K/V append slot
// and the guarded no-op on a geometry mismatch are all identical to
// make_qwen_attn_prepare_kernel(true, /*implicit_position_ramp=*/true), so the
// two realizations agree element for element. The position is always the
// implicit ramp pos0+t (that is what the engine records), and there is no
// positions buffer argument at all.
inline constexpr uint32_t qwen35_attn_model_head_dim = 256u;
inline constexpr uint32_t qwen35_attn_model_rope_dim = 64u;
inline constexpr uint32_t qwen35_attn_model_q_heads = 16u;
inline constexpr uint32_t qwen35_attn_model_kv_heads = 4u;
using QwenAttnPrepareModelKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q_proj [T, 2*q_heads*hd]*/,
    compute::Buffer<float> /*k_proj [T, kv_heads*hd]*/,
    compute::Buffer<float> /*v_proj [T, kv_heads*hd]*/,
    compute::Buffer<float> /*q_norm_w [hd]*/,
    compute::Buffer<float> /*k_norm_w [hd]*/,
    compute::Buffer<float> /*q out [T, q_heads, hd]*/,
    compute::Buffer<float> /*gate out [T, q_heads, hd]*/,
    compute::Buffer<float> /*k out [T, kv_heads, hd]*/,
    compute::Buffer<float> /*v out [T, kv_heads, hd]*/,
    compute::Buffer<float> /*k_cache*/,
    compute::Buffer<float> /*v_cache*/,
    uint32_t /*tokens*/, float /*theta*/, float /*eps*/,
    uint32_t /*pos0*/, uint32_t /*cache_capacity*/)>;
[[nodiscard]] QwenAttnPrepareModelKernel make_qwen35_attn_prepare_model_kernel(
    bool append_kv = true, QwenKvStorage kv = QwenKvStorage{}) noexcept;
// Device-state realization (static graph body): pos0 comes from
// state[qwen_dec_state_position]. Same appendix, same arithmetic.
using QwenAttnPrepareModelDevKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q_proj*/,
    compute::Buffer<float> /*k_proj*/,
    compute::Buffer<float> /*v_proj*/,
    compute::Buffer<float> /*q_norm_w*/,
    compute::Buffer<float> /*k_norm_w*/,
    compute::Buffer<float> /*q out*/,
    compute::Buffer<float> /*gate out*/,
    compute::Buffer<float> /*k out*/,
    compute::Buffer<float> /*v out*/,
    compute::Buffer<float> /*k_cache*/,
    compute::Buffer<float> /*v_cache*/,
    compute::Buffer<uint32_t> /*state*/,
    uint32_t /*tokens*/, float /*theta*/, float /*eps*/,
    uint32_t /*cache_capacity*/)>;
[[nodiscard]] QwenAttnPrepareModelDevKernel make_qwen35_attn_prepare_model_dev_kernel(
    bool append_kv = true, QwenKvStorage kv = QwenKvStorage{}) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_attn_prepare_model_dispatch(uint64_t tokens);
[[nodiscard]] Status validate_qwen35_attn_prepare_model_inputs(
    uint64_t tokens, const QwenDslConfig &cfg, uint64_t pos0, uint64_t cache_capacity);

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

// PACKED KV append (BF16 KV cache, see QwenKvStorage above): one thread per
// (token, kv_head, WORD) packs the `elements_per_word` stored codes of that word
// from the source row [T, kv_heads*head_dim] and writes one word at the dtype/layout
// address. Both codes are produced by the SAME thread, so no read-modify-write and
// no cross-lane race exists; at f32 (elements_per_word == 1) this reduces exactly to
// the dense append above. `pos0` is ignored by the device-state variant, which takes
// the absolute position of token 0 from state[qwen_dec_state_position] (so a captured
// CUDA graph stays argument-invariant, opt_plan/03 section 2.3).
using QwenKvAppendPackedKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*k [T, kv_heads*hd]*/, compute::Buffer<float> /*v [T, kv_heads*hd]*/,
    compute::Buffer<uint32_t> /*k_cache words*/, compute::Buffer<uint32_t> /*v_cache words*/,
    compute::Buffer<uint32_t> /*state [4] (device-state variant; dummy otherwise)*/,
    uint32_t /*tokens*/, uint32_t /*kv_heads*/, uint32_t /*head_dim*/,
    uint32_t /*pos0*/, uint32_t /*cache_capacity*/)>;
[[nodiscard]] QwenKvAppendPackedKernel make_qwen_kv_append_packed_kernel(
    QwenKvStorage kv, bool device_state = false) noexcept;
// One thread per word; `work_items` = tokens * kv_heads * ceil(head_dim/elements_per_word).
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_kv_append_packed_dispatch(
    uint64_t tokens, uint64_t kv_heads, uint64_t head_dim, QwenKvStorage kv);

// ---------------------------------------------------------------------------
// FP8 E4M3 KV append with ONLINE per-(head, 64-position block) scales.
//
// This is the only kernel that may write an fp8 cache. One 128-thread block per
// (64-position block, kv_head), K and V processed one after the other in the
// same block:
//
//  1. every thread reduces max|v| over ITS words of the NEW rows only (the rows
//     [max(pos0, block start), min(pos0+tokens, block end)) of the source
//     [tokens, kv_heads*head_dim]) -- non-finite values are excluded so one inf
//     cannot zero the block; the block maximum is then reduced through shared
//     memory (warp butterfly + one shared slot per warp);
//  2. thread 0 reads the block's CURRENT scale and decides: the new scale is
//     max(old_scale, max_abs/448), so `grow` is exactly "max_abs > 448*old_scale"
//     and a block that does not grow keeps its scale AND every earlier byte;
//  3. a grown block REWRITES its whole valid prefix: each earlier code is
//     dequantized with the OLD scale and re-encoded against the NEW one
//     (value/scale), so no old code is ever left paired with a new scale. The
//     ratio form value_new = code*old_scale/new_scale is exactly that.
//
// Race freedom: each thread owns whole 32-bit WORDS (four consecutive channels),
// so the four codes of a word are written by one thread and no read-modify-write
// crosses lanes; the scale store happens before the barrier that precedes the
// rewrite. Positions after the last valid one in the block are never touched.
//
// Cost (honest): the common decode case writes one row per (head, block) and
// reads the source twice (max pass + encode pass); a scale- GROWING append also
// re-reads and re-writes up to 64 rows of the block (16 KiB per head/K-V). That
// is the price of never storing an FP32 shadow cache and it is reported as
// QwenEngineInfo::kv_scaled_append. `pos0` is ignored by the device-state
// variant (position from state[qwen_dec_state_position]).
using QwenKvAppendFp8Kernel = compute::Kernel1D<void(
    compute::Buffer<float> /*k [tokens, kv_heads*head_dim]*/,
    compute::Buffer<float> /*v [tokens, kv_heads*head_dim]*/,
    compute::Buffer<uint32_t> /*k_cache words*/,
    compute::Buffer<uint32_t> /*v_cache words*/,
    compute::Buffer<float> /*k_scale [kv_heads*position_blocks]*/,
    compute::Buffer<float> /*v_scale [kv_heads*position_blocks]*/,
    compute::Buffer<uint32_t> /*state [4] (device-state variant; dummy otherwise)*/,
    uint32_t /*tokens*/, uint32_t /*kv_heads*/, uint32_t /*head_dim*/,
    uint32_t /*pos0*/, uint32_t /*cache_capacity*/)>;
[[nodiscard]] QwenKvAppendFp8Kernel make_qwen_kv_append_fp8_kernel(
    QwenKvStorage kv, bool device_state = false) noexcept;
// One 128-thread block per (touched 64-position block, kv_head);
// work_items = blocks_touched * kv_heads with blocks_touched =
// ceil((pos0 mod 64 + tokens) / 64). Fails closed on a non-fp8 storage (this
// kernel implements the SCALED contract only) or on an unusable geometry.
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_kv_append_fp8_dispatch(
    uint64_t pos0, uint64_t tokens, uint64_t kv_heads, QwenKvStorage kv);
// Shared-memory bytes one block of the fp8 append declares (the max reduction).
[[nodiscard]] uint32_t qwen_kv_append_fp8_shared_bytes() noexcept;

// ---------------------------------------------------------------------------
// Device halves of the fp8 KV codec. The append kernel above uses the same two
// helpers internally; these kernels expose them so the device codec can be
// compared bit-for-bit against qwen::f32_to_fp8_e4m3 / fp8_e4m3_to_f32 over the
// whole 256-code space and a dense value sweep. Test/diagnostic surface only --
// no engine path dispatches them.
// ---------------------------------------------------------------------------
using QwenKvFp8EncodeKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*values*/, compute::Buffer<uint32_t> /*codes*/, uint32_t /*count*/)>;
[[nodiscard]] QwenKvFp8EncodeKernel make_qwen_kv_fp8_encode_kernel() noexcept;
using QwenKvFp8DecodeKernel = compute::Kernel1D<void(
    compute::Buffer<uint32_t> /*codes*/, compute::Buffer<float> /*values*/, uint32_t /*count*/)>;
[[nodiscard]] QwenKvFp8DecodeKernel make_qwen_kv_fp8_decode_kernel() noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_kv_fp8_codec_dispatch(uint64_t count);

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
    compute::Buffer<float> /*k_scale (fp8: per (head,64-block); dummy otherwise)*/,
    compute::Buffer<float> /*v_scale (fp8: per (head,64-block); dummy otherwise)*/,
    compute::Buffer<float> /*out [T,q_heads,hd]*/,
    uint32_t /*tokens*/, uint32_t /*q_heads*/, uint32_t /*kv_heads*/,
    uint32_t /*head_dim*/, float /*scale*/, uint32_t /*pos0*/, uint32_t /*kv_len*/,
    uint32_t /*capacity*/)>;
[[nodiscard]] QwenAttnPrefillKernel make_qwen_attn_prefill_kernel(
    QwenKvStorage kv = QwenKvStorage{}) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_prefill_dispatch(uint64_t tokens, uint64_t q_heads);
[[nodiscard]] Status validate_qwen_attn_prefill_inputs(uint64_t tokens, uint64_t q_heads, uint64_t kv_heads,
                                                       uint64_t head_dim, uint64_t pos0, uint64_t kv_len,
                                                       uint64_t capacity);

// ---------------------------------------------------------------------------
// Model-path tiled causal GQA prefill attention (opt_plan/02 section 2, M-b).
//
// Replaces the untiled one-thread-per-(token,q_head) kernel above on the model
// path with a FlashAttention-style shared-memory tiled kernel. The defect it
// removes is structural: the legacy kernel indexes `Local<float>{1024}` with a
// RUNTIME head_dim, so every thread keeps a 4 KiB dynamic local array and every
// visited key re-reads and re-writes 2 x head_dim + 2 x head_dim words of local
// memory. Here head_dim, q_heads and kv_heads are COMPILE-TIME constants and
// every array extent and index is a compile-time constant, so all accumulators
// are register-allocated.
//
// One 128-thread block per (query tile, kv head). The block owns 4 warps =
// q_heads_per_kv warps, so every q head of one KV head shares the same K/V
// tiles (GQA dedup) and one shared-memory tile load serves 4 x q_tile
// (row, q_head) pairs. Lane l owns head-dim channels {l + 32c, c < head_dim/32}
// (bank-conflict-free strided access); Q, the online-softmax statistics and the
// output accumulator of its q_tile rows live in registers.
//
// Per key tile: cooperative coalesced K/V staging -> warp all-reduce
// (warp_active_sum) scores -> causal mask -> tile max -> ONE rescale of the
// accumulator per row -> P.V fused into the same accumulators -> block barrier.
// The online FP32 softmax keeps the reference kernel's running max/sum/output
// contract, so results differ only by FP32 reassociation.
//
// The kernel is genuinely dynamic-length: tokens, pos0, kv_len and capacity stay
// runtime scalar arguments, so one compiled shader serves every prompt length,
// position, ragged tail, KV capacity and defensive clamp (no per-position JIT).
// ---------------------------------------------------------------------------
struct QwenAttnPrefillTiledConfig {
    // Compile-time model geometry (Qwen3.5-4B full attention).
    uint32_t head_dim{256};// head_dim % 32 == 0; head_dim/32 registers per lane per row
    uint32_t q_heads{16};
    uint32_t kv_heads{4};// q_heads % kv_heads == 0; q_heads / kv_heads == warps per block
    // Tiling. q_tile bounds the per-thread accumulators (q_tile * head_dim/32
    // floats for Q and the same again for the output); k_tile bounds the shared
    // K/V staging: 2 * k_tile * head_dim * 4 B.
    //
    // The defaults are MEASURED on the target box (RTX 4080 SUPER, CUDA), not
    // taken from the plan's 32/64 sketch: the unrolled per-tile body already
    // reaches the 255-register ceiling at these extents, which caps occupancy at
    // 8 warps/SM (Nsight Compute: 16.7% theoretical, 0.28 issued warps per
    // scheduler). Growing q_tile or k_tile only adds live state and loses
    // occupancy, so the measured optimum is the SMALLEST tile that still
    // amortizes the staging: S=1024 wall time 2.32 ms (q_tile 2, k_tile 2) vs
    // 2.80 ms (4/2), 2.98 ms (4/8) and 3.49 ms (4/16). See
    // .kimix_cache/qwen_benchmark/phase2/ for the sweep.
    uint32_t q_tile{2};
    uint32_t k_tile{2};
};
// Shared-memory bytes one block of this configuration declares (K + V tiles).
[[nodiscard]] uint32_t qwen35_attn_prefill_tiled_shared_bytes(const QwenAttnPrefillTiledConfig &cfg) noexcept;
[[nodiscard]] Status validate_qwen35_attn_prefill_tiled_config(const QwenAttnPrefillTiledConfig &cfg) noexcept;
// Pick the tiling for a device WITHOUT touching a Device: the caller passes the
// reported warp size and compute_max_shared_memory_size() (0 = the backend does
// not report one, e.g. DX, which then gets the portable 16 KiB default). Fails
// closed when the kernel cannot run (warp size != 32, shared budget below the
// smallest tile), so a caller can keep the reference kernel instead.
[[nodiscard]] Status select_qwen35_attn_prefill_tiled_config(uint32_t warp_size,
                                                             uint32_t max_shared_memory_bytes,
                                                             QwenAttnPrefillTiledConfig &cfg) noexcept;
// q/gate/out are [tokens, q_heads, head_dim]; caches are
// [capacity, kv_heads, head_dim]; cache layout and output/gate layout are
// identical to the reference kernel above. head_dim/q_heads/kv_heads are
// compile-time (cfg); tokens/scale/pos0/kv_len/capacity stay runtime.
using QwenAttnPrefillTiledKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q [T,q_heads,hd]*/,
    compute::Buffer<float> /*gate [T,q_heads,hd]*/,
    compute::Buffer<float> /*k_cache [capacity,kv_heads,hd]*/,
    compute::Buffer<float> /*v_cache [capacity,kv_heads,hd]*/,
    compute::Buffer<float> /*k_scale (fp8: per (head,64-block); dummy otherwise)*/,
    compute::Buffer<float> /*v_scale (fp8: per (head,64-block); dummy otherwise)*/,
    compute::Buffer<float> /*out [T,q_heads,hd]*/,
    uint32_t /*tokens*/, float /*scale*/, uint32_t /*pos0*/, uint32_t /*kv_len*/,
    uint32_t /*capacity*/)>;
// Pins warp size 32; the host must check device.compute_warp_size() == 32.
[[nodiscard]] QwenAttnPrefillTiledKernel make_qwen35_attn_prefill_tiled_kernel(
    const QwenAttnPrefillTiledConfig &cfg,
    QwenKvStorage kv = QwenKvStorage{}) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_attn_prefill_tiled_dispatch(
    uint64_t tokens, const QwenAttnPrefillTiledConfig &cfg);
// Same input contract as validate_qwen_attn_prefill_inputs (this calls it), plus
// the tiled-configuration checks.
[[nodiscard]] Status validate_qwen35_attn_prefill_tiled_inputs(uint64_t tokens, uint64_t pos0,
                                                               uint64_t kv_len, uint64_t capacity,
                                                               const QwenAttnPrefillTiledConfig &cfg);

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
    compute::Buffer<float> /*k_scale (fp8: per (head,64-block); dummy otherwise)*/,
    compute::Buffer<float> /*v_scale (fp8: per (head,64-block); dummy otherwise)*/,
    compute::Buffer<float> /*out [q_heads,hd]*/,
    uint32_t /*q_heads*/, uint32_t /*kv_heads*/, uint32_t /*head_dim*/,
    float /*scale*/, uint32_t /*kv_len*/, uint32_t /*capacity*/)>;
// Pins warp size 32; host must check device.compute_warp_size() == 32.
// `kv_len_limit` makes the kernel STATICALLY GUARDED (opt_plan/03 section 4.2):
// 0 = the reference behaviour, every kv_len >= 1 runs; N > 0 = every block
// returns immediately when kv_len >= N, i.e. the split-KV path owns that range.
// The limit is a compile-time constant baked into the kernel, while kv_len keeps
// coming from the host argument (or from the device state below), so a captured
// graph can hold BOTH realizations without a single per-token update.
[[nodiscard]] QwenAttnDecodeKernel make_qwen_attn_decode_kernel(
    uint32_t kv_len_limit = 0u, QwenKvStorage kv = QwenKvStorage{}) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_decode_dispatch(uint64_t q_heads);
[[nodiscard]] Status validate_qwen_attn_decode_inputs(uint64_t q_heads, uint64_t kv_heads,
                                                      uint64_t head_dim, uint64_t kv_len, uint64_t capacity);

// Device-state realization: the key count comes from
// state[qwen_dec_state_kv_len] (clamped to `capacity` exactly like the host
// argument) instead of a host scalar. Same block/warp mapping, same FP32 online
// softmax, same sigmoid gate: bit-identical output for the same kv_len.
using QwenAttnDecodeDevKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q [q_heads,hd]*/,
    compute::Buffer<float> /*gate [q_heads,hd]*/,
    compute::Buffer<float> /*k_cache*/, compute::Buffer<float> /*v_cache*/,
    compute::Buffer<float> /*k_scale (fp8: per (head,64-block); dummy otherwise)*/,
    compute::Buffer<float> /*v_scale (fp8: per (head,64-block); dummy otherwise)*/,
    compute::Buffer<float> /*out [q_heads,hd]*/,
    compute::Buffer<uint32_t> /*state [4]*/,
    uint32_t /*q_heads*/, uint32_t /*kv_heads*/, uint32_t /*head_dim*/,
    float /*scale*/, uint32_t /*capacity*/)>;
[[nodiscard]] QwenAttnDecodeDevKernel make_qwen_attn_decode_dev_kernel(
    uint32_t kv_len_limit = 0u, QwenKvStorage kv = QwenKvStorage{}) noexcept;

// ---------------------------------------------------------------------------
// Split-KV decode attention with GQA dedup (opt_plan/03 section 4.2).
//
// `make_qwen_attn_decode_kernel` above is one block PER q_head: each of the 4
// q_heads of a KV head re-streams the same K/V rows (4x read amplification) and
// the whole decode attention is 16 blocks, i.e. 0.2 blocks/SM on an 80-SM part.
// This family replaces it for long contexts:
//
//   one block per (kv_head, key_split), 128 threads = 4 warps, one warp per
//   q_head of that KV head. All 4 warps consume the SAME staged K/V tile
//   (a [k_tile, head_dim] K tile and the matching V tile in shared memory), so
//   every cache row is fetched ONCE per (layer, token) instead of four times.
//   Each warp runs its own FP32 online softmax over its split's key range and
//   writes (m, l, acc[head_dim]) to a partial buffer; a combine kernel reduces
//   the <= max_splits partials with the rescaling identity and applies the
//   sigmoid output gate.
//
// Contract that makes the static CUDA graph argument-invariant (03 section 2.3):
//   * the grid is kv_heads * max_splits blocks for EVERY token (max_splits is a
//     compile-time constant), and the SPLIT COUNT is derived in-kernel from the
//     same kv_len the legacy kernel reads, so no launch geometry changes;
//   * `threshold` is compile-time: the split family returns immediately when
//     kv_len < threshold and the legacy kernel returns immediately when
//     kv_len >= threshold, so the graph holds both and exactly one of them runs;
//   * splits beyond the selected count are EMPTY: they publish (m=sentinel,
//     l=0, acc=0) and the combine kernel skips them (l <= 0), so stale partials
//     of an earlier, longer step are never read;
//   * the partial buffers are FIXED at max_splits (host allocation) and every
//     index is bounds-checked at create, so no kv_len can overflow them.
//
// Splits: splits = clamp(ceil(kv_len / split_chunk), 1, max_splits) and split i
// owns the balanced range [i*per, min(kv_len, (i+1)*per)), per = ceil(kv_len /
// splits). That is the plan's "scale the split count with kv_len, capped by a
// device-sized bound" rule: a longer context keeps ~split_chunk keys per split
// until the cap binds, and beyond the cap each split simply gets more keys.
//
// Numerics: the FP32 partials change the summation ORDER relative to the legacy
// per-q_head merge, so the two agree to the documented FP32 reassociation
// tolerance (tests/integration/qwen/test_qwen_ops_dsl.cpp) rather than
// bit-exactly. `ref_qwen_attn_decode` stays the CPU oracle for both.
struct QwenAttnDecodeSplitConfig {
    // Compile-time model geometry. The Qwen3.5-4B text trunk is
    // q_heads 16 / kv_heads 4 / head_dim 256 (GQA 4, one warp per q head).
    uint32_t head_dim{256};
    uint32_t q_heads{16};
    uint32_t kv_heads{4};
    // Keys staged per shared-memory tile. Shared footprint =
    // 2 * k_tile * head_dim * 4 B (K tile + V tile), so k_tile 8 = 16 KiB,
    // 16 = 32 KiB, 32 = 64 KiB.
    //
    // MEASURED on the target box (RTX 4080 SUPER, 48 KiB reported static limit,
    // isolated single-query arm, k_tile 8 / 16 with max_splits 16):
    // kv_len 128: 0.0362 / 0.0374 ms; 2048: 0.1296 / 0.1343; 8192: 0.1393 /
    // 0.1475; 32768: 0.6636 / 0.7159. The SMALLER tile wins everywhere, because
    // the binding resource is occupancy, not staging: 16 KiB lets 3 blocks/SM
    // co-reside inside a 48 KiB budget while 32 KiB allows one, and the four q
    // heads of a KV head already amortize the staging 4x. k_tile 32 (64 KiB)
    // does not fit a 48 KiB static budget at all (the driver rejects the image).
    // So select_... keeps the measured 16 KiB tile on a device that reports no limit
    // (DX) or one >= 16 KiB (the target's 48 KiB); k_tile 16 remains available for a
    // device whose 96 KiB+ budget makes the trade different. On a SMALLER reported
    // limit select_... halves the tile until the staging fits (4 KiB -> 2 keys, 8/12
    // KiB -> 4 keys), because the chosen tile must never exceed the budget.
    uint32_t k_tile{8};
    // Keys per split before the max_splits cap binds.
    uint32_t split_chunk{512};
    // Compile-time grid bound: blocks = kv_heads * max_splits. This 16 is only the
    // plain-struct placeholder: select_... always overwrites it with
    // qwen_attn_decode_split_bound, which is the documented 40 on a backend that
    // cannot report an SM count (see qwen_attn_decode_split_default_bound).
    uint32_t max_splits{16};
    // kv_len below which the legacy per-q_head kernel owns the step.
    uint32_t threshold{512};
};
// Shared-memory bytes one block of this configuration declares (K tile + V tile).
[[nodiscard]] uint32_t qwen_attn_decode_split_shared_bytes(const QwenAttnDecodeSplitConfig &cfg) noexcept;
// Bytes of the two fixed partial buffers: max_splits*q_heads*head_dim floats
// (acc) and max_splits*q_heads*2 floats (m, l).
[[nodiscard]] uint64_t qwen_attn_decode_split_partial_bytes(const QwenAttnDecodeSplitConfig &cfg) noexcept;
[[nodiscard]] Status validate_qwen_attn_decode_split_config(const QwenAttnDecodeSplitConfig &cfg) noexcept;
  // Split-count bound derived from a REAL SM count when the caller has one, or the
  // documented fail-closed default when the backend exposes none.
  // sm_count != 0 -> clamp(qwen_attn_decode_split_blocks_per_sm * sm_count /
  // kv_heads, 1, qwen_attn_decode_split_cap)
  // sm_count == 0 -> qwen_attn_decode_split_default_bound (40), the documented
  // MEASURED default (see qwen_attn_decode_split_blocks_per_sm)
  // -- used when the backend answers no SM count (DX/VK; see qwen/caps.h).
  // The caller is expected to pass QwenBackendCaps::sm_count. CUDA answers it with
  // the driver's multiprocessor count (patches/luisa-cuda-multiprocessor-query.patch);
  // 0 means "not queryable" there (driver failure), and this function then reports
  // the DEFAULT, never a guessed device property. On an 80-SM part the derived bound
  // is 2 * 80 / 4 = 40 = the cap, i.e. identical to the default, so adding the query
  // changes no grid on the target box.
  inline constexpr uint32_t qwen_attn_decode_split_cap = 40u;
// Measured optimum of the block count per SM, from the isolated single-query arm
// on the target box (80 SMs, 4 kv heads, k_tile 8, kv_len 16384/32768):
//   max_splits 16 (64 blocks, 0.8/SM): 396 / 408 GB/s
//   max_splits 20 (80 blocks, 1.0/SM): 473 / 488 GB/s
//   max_splits 24 (96 blocks, 1.2/SM): 514 / 534 GB/s
//   max_splits 40 (160 blocks, 2.0/SM): 589 / 639 GB/s   (= the 634 GB/s roof)
// 2 blocks/SM is the measured knee and it costs < 1 % of an end-to-end token at
// kv_len <= 2048 (where the static grid is mostly no-op blocks; measured 0.11 ms
// per token at kv_len 512). The splits actually used still scale with kv_len
// (ceil(kv_len / split_chunk), so 16 at kv_len 8192 and 40 at 32768), which keeps
// the T16 band at 16-20 splits and only reaches the plan's T32 cap of 40 for
// long contexts.
inline constexpr uint32_t qwen_attn_decode_split_blocks_per_sm = 2u;
inline constexpr uint32_t qwen_attn_decode_split_default_bound = 40u;
[[nodiscard]] uint32_t qwen_attn_decode_split_bound(uint32_t sm_count, uint32_t kv_heads) noexcept;
// Pick k_tile from the REPORTED shared-memory limit (0 = "backend reports none",
// e.g. DX, which then gets the portable 16 KiB tile) and max_splits from the
// split bound. k_tile is the measured optimum (8 keys = 16 KiB) whenever the limit
// allows it and otherwise the largest power of two below it that fits
// (2 KiB -> 1 key, 4 KiB -> 2, 8/12 KiB -> 4, >= 16 KiB -> 8), so a nonzero limit is
// ALWAYS honoured: the caller never receives a tile over its budget.
// Fails closed on a non-32 warp size or a budget below the one-key floor
// (qwen_attn_decode_split_shared_bytes of k_tile 1 = 2 KiB at head_dim 256).
[[nodiscard]] Status select_qwen_attn_decode_split_config(uint32_t warp_size,
                                                          uint32_t max_shared_memory_bytes,
                                                          uint32_t sm_count,
                                                          QwenAttnDecodeSplitConfig &cfg) noexcept;
// Gate for a caller override of the compile-time split bound
// (QwenEngineOptions::dsl_split_kv_max_splits / --split-kv-max-splits).
//   max_splits == 0 -> no-op success, the derived bound is kept.
//   max_splits > qwen_attn_decode_split_cap -> refused.
//   otherwise the config is re-validated (the uint32 partial-workspace overflow gate)
//   and its K/V tile is re-checked against `max_shared_memory_bytes`.
// On refusal `cfg` is left untouched and the returned status is fail-closed: the
// engine then keeps the per-q_head kernel instead of compiling a grid it cannot
// index or a tile the driver would reject.
[[nodiscard]] Status qwen_attn_decode_split_apply_max_splits_override(
    QwenAttnDecodeSplitConfig &cfg, uint32_t max_splits,
    uint32_t max_shared_memory_bytes) noexcept;
// Host mirrors of the in-kernel split rule (tests and the byte model use them;
// the kernels recompute the same quantities from kv_len so no scalar is needed).
[[nodiscard]] uint32_t qwen_attn_decode_split_count(uint64_t kv_len,
                                                    const QwenAttnDecodeSplitConfig &cfg) noexcept;
// Keys split `split` owns at this kv_len (0 = empty split).
[[nodiscard]] uint32_t qwen_attn_decode_split_span(uint64_t kv_len, uint32_t split,
                                                   const QwenAttnDecodeSplitConfig &cfg) noexcept;
// Blocks the split kernel launches: kv_heads * max_splits, independent of kv_len.
[[nodiscard]] uint32_t qwen_attn_decode_split_blocks(const QwenAttnDecodeSplitConfig &cfg) noexcept;

// q [q_heads,hd]; caches [capacity,kv_heads,hd]; partial_acc
// [max_splits,q_heads,hd]; partial_ml [max_splits,q_heads,2] = (m, l). Writes the
// partials of its (kv_head, split) only, and only while kv_len >= threshold.
// Caps and split bounds are compile-time; kv_len/scale/capacity stay runtime.
using QwenAttnDecodeSplitKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q [q_heads,hd]*/,
    compute::Buffer<float> /*k_cache*/, compute::Buffer<float> /*v_cache*/,
    compute::Buffer<float> /*k_scale (fp8: per (head,64-block); dummy otherwise)*/,
    compute::Buffer<float> /*v_scale (fp8: per (head,64-block); dummy otherwise)*/,
    compute::Buffer<float> /*partial_acc*/, compute::Buffer<float> /*partial_ml*/,
    uint32_t /*kv_len*/, float /*scale*/, uint32_t /*capacity*/)>;
[[nodiscard]] QwenAttnDecodeSplitKernel make_qwen_attn_decode_split_kernel(
    const QwenAttnDecodeSplitConfig &cfg,
    QwenKvStorage kv = QwenKvStorage{}) noexcept;
// Device-state realization: kv_len from state[qwen_dec_state_kv_len].
using QwenAttnDecodeSplitDevKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*q [q_heads,hd]*/,
    compute::Buffer<float> /*k_cache*/, compute::Buffer<float> /*v_cache*/,
    compute::Buffer<float> /*k_scale (fp8: per (head,64-block); dummy otherwise)*/,
    compute::Buffer<float> /*v_scale (fp8: per (head,64-block); dummy otherwise)*/,
    compute::Buffer<float> /*partial_acc*/, compute::Buffer<float> /*partial_ml*/,
    compute::Buffer<uint32_t> /*state [4]*/,
    float /*scale*/, uint32_t /*capacity*/)>;
[[nodiscard]] QwenAttnDecodeSplitDevKernel make_qwen_attn_decode_split_dev_kernel(
    const QwenAttnDecodeSplitConfig &cfg,
    QwenKvStorage kv = QwenKvStorage{}) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_decode_split_dispatch(
    const QwenAttnDecodeSplitConfig &cfg);
[[nodiscard]] Status validate_qwen_attn_decode_split_inputs(uint64_t kv_len, uint64_t capacity,
                                                            const QwenAttnDecodeSplitConfig &cfg);
// Combine: one 128-thread block per q_head. Reduces the `splits` partials with the
// rescaling identity, normalizes by the softmax denominator, multiplies by the
// sigmoid gate and writes out [q_heads,hd]. Skips empty splits (l <= 0), and a
// degenerate all-empty reduction writes a finite 0 instead of 0/0.
using QwenAttnDecodeCombineKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*partial_acc*/, compute::Buffer<float> /*partial_ml*/,
    compute::Buffer<float> /*gate [q_heads,hd]*/, compute::Buffer<float> /*out [q_heads,hd]*/,
    uint32_t /*kv_len*/, uint32_t /*capacity*/)>;
[[nodiscard]] QwenAttnDecodeCombineKernel make_qwen_attn_decode_combine_kernel(
    const QwenAttnDecodeSplitConfig &cfg) noexcept;
using QwenAttnDecodeCombineDevKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*partial_acc*/, compute::Buffer<float> /*partial_ml*/,
    compute::Buffer<float> /*gate [q_heads,hd]*/, compute::Buffer<float> /*out [q_heads,hd]*/,
    compute::Buffer<uint32_t> /*state [4]*/, uint32_t /*capacity*/)>;
[[nodiscard]] QwenAttnDecodeCombineDevKernel make_qwen_attn_decode_combine_dev_kernel(
    const QwenAttnDecodeSplitConfig &cfg) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_decode_combine_dispatch(
    const QwenAttnDecodeSplitConfig &cfg);

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
// Device-state realization (opt_plan/09 V3): the absolute position of the first
// token comes from state[qwen_dec_state_position] instead of a host scalar, so
// the 24 GDN layers of a static decode graph keep their command list unchanged
// while the conv ring still advances by one position per launch. Identical
// arithmetic and ring indexing; the ring writes stay strictly after the reads
// of the older taps in the same thread.
using QwenGdnConvDevKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*src [T,C] pre-activation qkv*/,
    compute::Buffer<float> /*weight [kernel,C] GGUF channel-major payload w[c*K+j]*/,
    compute::Buffer<float> /*ring [C,kernel-1] in/out*/,
    compute::Buffer<float> /*out [T,C]*/,
    compute::Buffer<uint32_t> /*state [4]*/,
    uint32_t /*tokens*/, uint32_t /*channels*/, uint32_t /*kernel_size*/)>;
[[nodiscard]] QwenGdnConvDevKernel make_qwen_gdn_conv_silu_dev_kernel() noexcept;
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
// 128-bit twin of the T=1 alpha/beta gate fusion (opt_plan/09 N2): uint4 weight
// loads, one scale per uint4, float4 activation loads and the branchless
// LUT-free E4M3 decode. Same (2*heads)-warp geometry, same gate epilogues.
using QwenGdnDecodeGatesPackedKernel = compute::Kernel1D<void(
    compute::Buffer<float4> /*x [K] as float4*/,
    compute::Buffer<uint4> /*alpha W [H,K] packed*/, compute::Buffer<float> /*alpha scales*/,
    compute::Buffer<uint4> /*beta W [H,K] packed*/, compute::Buffer<float> /*beta scales*/,
    compute::Buffer<float> /*ssm_a [H]*/, compute::Buffer<float> /*dt_bias [H]*/,
    compute::Buffer<float> /*beta out [H]*/, compute::Buffer<float> /*g out [H]*/,
    uint32_t /*K*/, uint32_t /*heads*/, uint32_t /*group*/, float /*softplus threshold*/,
    uint32_t /*vecs_per_row*/)>;
[[nodiscard]] QwenGdnDecodeGatesPackedKernel make_qwen_gdn_decode_gates_packed_kernel(
    QwenWeightFormat format) noexcept;
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_decode_gates_packed_dispatch(
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
// Fused decode realization (opt_plan/09 N3.3): conv4 + SiLU + ring update +
// QK L2 normalization for a SINGLE token. The unfused decode path writes the
// whole conv output (2*qk_heads*hd Q/K channels + v_heads*hd V channels), then
// a second launch re-reads the Q/K half to normalize it into _gdn_q/_gdn_k --
// 24 extra launches and one round trip of the Q/K half per GDN layer per token.
// Here one warp owns 128 consecutive conv channels (lane l owns channels
// base+l, base+l+32, ...): it computes the depthwise conv against the ring,
// writes its V channels straight to the conv output (which is what the
// recurrent kernel consumes) and, for the Q and K channel groups, reduces the
// sum of squares with warp_active_sum and writes the normalized (+ Q-scaled)
// values directly. Composition and order are identical to the unfused pair, so
// the Q/K outputs are numerically identical (the FLA/HF l2norm keeps eps INSIDE
// the rsqrt and q is scaled by 1/sqrt(hd)); only the intermediate is removed.
// head_dim is the compile-time model geometry (128); any other head dimension is
// a guarded no-op, and the host planner refuses it. tokens == 1 only.
using QwenGdnDecodeFusedKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*qkv [1, channels] packed projection*/,
    compute::Buffer<float> /*conv_w [channels, kernel] (channel-major GGUF)*/,
    compute::Buffer<float> /*ring [channels, kernel-1] in/out*/,
    compute::Buffer<float> /*q_out [qk_heads, hd] normalized + q_scale*/,
    compute::Buffer<float> /*k_out [qk_heads, hd] normalized*/,
    compute::Buffer<float> /*conv_out [1, channels] (V channels written)*/,
    uint32_t /*qk_heads*/, uint32_t /*head_dim*/, uint32_t /*v_channels*/,
    uint32_t /*kernel_size*/, float /*eps*/, float /*q_scale*/, uint32_t /*pos0*/)>;
using QwenGdnDecodeFusedDevKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*qkv*/, compute::Buffer<float> /*conv_w*/,
    compute::Buffer<float> /*ring*/, compute::Buffer<float> /*q_out*/,
    compute::Buffer<float> /*k_out*/, compute::Buffer<float> /*conv_out*/,
    compute::Buffer<uint32_t> /*state (position)*/,
    uint32_t /*qk_heads*/, uint32_t /*head_dim*/, uint32_t /*v_channels*/,
    uint32_t /*kernel_size*/, float /*eps*/, float /*q_scale*/)>;
[[nodiscard]] QwenGdnDecodeFusedKernel make_qwen35_gdn_decode_fused_kernel() noexcept;
// Device-state realization (identical numerics; reads the absolute position from
// _dev_state so the 24 GDN layers of a static decode graph stay
// argument-invariant, opt_plan/09 V3).
[[nodiscard]] QwenGdnDecodeFusedDevKernel make_qwen35_gdn_decode_fused_dev_kernel() noexcept;
// One 128-thread block per 512 conv channels (4 warps, one warp per head row);
// `work_items` is the block count. Fails closed when the channel count is not a
// multiple of 512 or the geometry leaves an uncovered channel.
[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_gdn_decode_fused_dispatch(
    uint64_t qk_heads, uint64_t head_dim, uint64_t v_channels);
[[nodiscard]] Status validate_qwen35_gdn_decode_fused_inputs(
    uint64_t tokens, uint64_t qk_heads, uint64_t head_dim, uint64_t v_channels,
    uint64_t kernel_size, uint64_t pos0);

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
  // The Qwen3.5-4B GDN head width, the only geometry the two model
  // specializations below compile for (opt_plan/09 V4).
  inline constexpr uint32_t qwen35_gdn_model_head_dim = 128u;
  [[nodiscard]] QwenGdnRecurrentKernel make_qwen_gdn_recurrent_kernel() noexcept;
  // Model specialization of the per-token recurrence with a COMPILE-TIME head
  // dimension (same reasoning as make_qwen35_gdn_chunk_kernel below: the
  // lane-owned state column and every key-axis index become constants, so the
  // 128-float column is register-resident instead of a runtime-indexed local
  // array). Only head_dim == 128 is specialized; anything else returns the
  // generic kernel, and the specialized kernel guards head_dim == 128 itself.
  [[nodiscard]] QwenGdnRecurrentKernel make_qwen35_gdn_recurrent_kernel(
      uint32_t head_dim = qwen35_gdn_model_head_dim) noexcept;
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
  // Model specialization (opt_plan/09 V4, opt_plan/08 N3.4): the same WY/UT
  // recurrence with a COMPILE-TIME head dimension, so the lane-owned state
  // column `col[head_dim]` and every key-axis loop index are constants. The
  // generic factory above indexes that Local<> array with a runtime Var<uint>,
  // which cannot be proven constant and therefore lives in local memory across
  // the whole chunk loop. Only head_dim == 128 (the Qwen3.5-4B GDN geometry) is
  // specialized; every other value returns the generic kernel. Identical FP32
  // math and accumulation order: only the extent/indices become compile-time.
  // The kernel additionally guards `head_dim == 128` at run time, so binding a
  // different geometry through this factory is a no-op instead of a misread.
  [[nodiscard]] QwenGdnChunkKernel make_qwen35_gdn_chunk_kernel(
      uint32_t head_dim = qwen35_gdn_model_head_dim) noexcept;
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
  // FUSED per-token GDN recurrence + gated output RMSNorm (opt_plan/03 section 5,
  // the one-CTA arm of the "GDN recurrent + gated norm" row).
  //
  // The decode schedule has three parts: the recurrence (one launch), the gated
  // output RMSNorm (a second launch, because it needs a reduction over the whole
  // head) and the output projection. The 4-CTA-per-head decode kernel
  // (`make_qwen35_gdn_decode4_kernel`) cannot own that reduction -- its four
  // blocks of one head are independent -- so it must round-trip the head row
  // through global memory (`_gdn_out`) and pay the second launch.
  //
  // This factory fuses BOTH into ONE launch on the ONE-CTA-per-head recurrence,
  // where the whole head lives in one block:
  //   * the state update, the err/output dots and their FP32 accumulation order
  //     are the ones of `make_qwen35_gdn_recurrent_kernel` head_dim=128 (same
  //     body, same per-lane serial reduction over the key axis);
  //   * because one lane owns exactly one value column, the gated RMSNorm is a
  //     reduction over DISTINCT registers and needs no staging at all: per-lane
  //     `o*o` -> `warp_active_sum` (the partial of channels [32w, 32w+32)) -> one
  //     sync_block() -> a fixed 4-term fold over the warp partials that every lane
  //     performs identically, so `inv` is bit-identical across the four warps and
  //     no broadcast is needed. That is the block-wide reduction the 4-CTA kernel
  //     cannot do, since its four blocks hold disjoint key slices;
  //   * the DIRECT norm weight and the SiLU(z) gate are applied in the same block,
  //     so neither the recurrent output nor the normalized output ever touches
  //     global memory: no `_gdn_out` round trip, no second launch.
  //
  // Consequence: the same recurrence as the one-CTA pair and the same norm formula,
  // with the head reduction re-associated (32-lane warp trees + a 4-way fold instead
  // of four strided per-lane sums on one warp). The differential test compares BOTH
  // against the FP64 oracle and pins the pair deviation at the FP32 last-bit
  // envelope (measured <= 3.8e-06 absolute over six carried decode steps). tokens == 1 only and
  // head_dim == 128 (the model geometry); anything else is a guarded no-op and the
  // host planner/validator refuse it.
  using QwenGdnRecurrentNormKernel = compute::Kernel1D<void(
      compute::Buffer<float> /*q [1,qk_heads,hd] normalized+scaled*/,
      compute::Buffer<float> /*k [1,qk_heads,hd] normalized*/,
      compute::Buffer<float> /*conv_out [1, qkv_channels] (V source)*/,
      compute::Buffer<float> /*beta [1,v_heads]*/,
      compute::Buffer<float> /*g [1,v_heads]*/,
      compute::Buffer<float> /*state [v_heads,hd,hd] in/out*/,
      compute::Buffer<float> /*z [1,v_heads*hd] (attention_gate projection)*/,
      compute::Buffer<float> /*norm_w [hd] (DIRECT weight)*/,
      compute::Buffer<float> /*out [1,v_heads,hd] gated RMSNorm result*/,
      uint32_t /*tokens (must be 1)*/, uint32_t /*v_heads*/, uint32_t /*head_dim (must be 128)*/,
      uint32_t /*qk_heads*/, uint32_t /*v_offset (2*qk_heads*hd)*/, float /*eps*/)>;
  // Model geometry only (head_dim 128, one 128-thread block per V head). The kernel
  // guards `tokens == 1 && head_dim == 128` itself, so binding another geometry is a
  // no-op instead of a misread; plan/validate refuse it up front.
  [[nodiscard]] QwenGdnRecurrentNormKernel make_qwen35_gdn_recurrent_norm_kernel() noexcept;
  // One block per V head (v_heads blocks of 128 threads).
  [[nodiscard]] Result<QwenLaunch1D> plan_qwen35_gdn_recurrent_norm_dispatch(uint64_t v_heads);
  [[nodiscard]] Status validate_qwen35_gdn_recurrent_norm_inputs(
      uint64_t tokens, uint64_t v_heads, uint64_t qk_heads, uint64_t head_dim);
  // Shared-memory bytes per block: staged k, staged q and the four warp partials
  // of the norm reduction (2 * 128 + 4 floats for the model geometry).
  [[nodiscard]] uint64_t qwen_gdn_recurrent_norm_shared_bytes(uint64_t head_dim) noexcept;

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
