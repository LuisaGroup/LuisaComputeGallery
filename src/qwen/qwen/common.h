#pragma once
// Qwen3.5-4B shared engine layer -- common host contract.
// Host-only: no DSL/runtime device includes in this header.
//
// Contents:
// - Recoverable Status/Result for the whole qwen layer. This deliberately
//   REUSES luisa::ocr::Status/Result (src/ocr/ocr/status.h) so the gallery
//   keeps a single recoverable-error contract; the aliases below make qwen
//   code read naturally without depending on any other OCR component.
// - Checked 64-bit host arithmetic (add/mul/align/element-count) shared by
//   the GGUF parser and the weight packers. Overflow is always a recoverable
//   failure, never UB.
// - Explicit bit-exact BF16 (RNE), FP8 E4M3FN and IEEE FP16 codecs plus the
//   storage-ABI packing helpers:
//     * BF16 tensors: two BF16 code units per little-endian uint32_t word,
//       LOW element first (raw copy of the GGUF BF16 bytes on LE hosts).
//     * FP8 E4M3 tensors: four E4M3 codes per little-endian uint32_t word,
//       LOW element first, FP32 scale per contiguous group of 64 K elements
//       per output row, scale = max_abs / 448. The codes are CONVERTED at load
//       time from the BF16 source and the conversion is reported.
//     * Q8_0 tensors: four SIGNED int8 codes per little-endian uint32_t word,
//       LOW element first -- a byte-for-byte copy of the GGUF block payload --
//       plus one exact FP16->FP32 widened scale per contiguous group of 32 K
//       elements per output row. NOTHING is re-quantized: the device words are
//       the checkpoint's own int8 block codes.
//   FP8 here means E4M3FN (finite, max 448, NaN code 0x7f/0xff) -- never Q8_0,
//   and Q8_0 is never routed through FP8.
// - Weight storage classification helpers (which tensors stay dense F32).
//
// Library code never throws and never uses RTTI. Ordinary failures (bad
// files, unsupported formats, overflow, IO errors) are reported as Status.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/status.h"

namespace luisa::qwen {

// Recoverable status contract shared with the OCR engine (see header note).
using ocr::Diagnostic;
using ocr::DiagnosticSeverity;
using ocr::Result;
using ocr::Status;
using ocr::StatusCode;
using ocr::diagnostic_severity_name;
using ocr::status_code_name;

// ---------------------------------------------------------------------------
// Checked 64-bit host arithmetic. On overflow these return false and leave
// `out` untouched.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool checked_add(uint64_t a, uint64_t b, uint64_t &out) noexcept {
    if (b > UINT64_MAX - a) { return false; }
    out = a + b;
    return true;
}
[[nodiscard]] inline bool checked_mul(uint64_t a, uint64_t b, uint64_t &out) noexcept {
    if (a != 0 && b > UINT64_MAX / a) { return false; }
    out = a * b;
    return true;
}
// Align `value` up to `alignment` (power of two); fails on overflow or a
// non-power-of-two alignment.
[[nodiscard]] inline bool checked_align_up(uint64_t value, uint64_t alignment, uint64_t &out) noexcept {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) { return false; }
    const auto mask = alignment - 1;
    if (value > UINT64_MAX - mask) { return false; }
    out = (value + mask) & ~mask;
    return true;
}
// Product of all dimensions (checked). An empty shape yields 1 (scalar).
[[nodiscard]] bool checked_element_count(luisa::span<const uint64_t> dims, uint64_t &out) noexcept;

// ---------------------------------------------------------------------------
// BF16 codec (bit-exact, round-to-nearest-even; no hardware rounding mode).
// NaN inputs produce a quiet BF16 NaN; infinities are preserved; FP32 values
// that overflow BF16's range become +/-inf (IEEE RNE).
// ---------------------------------------------------------------------------
[[nodiscard]] float bf16_to_f32(uint16_t b) noexcept;
[[nodiscard]] uint16_t f32_to_bf16(float f) noexcept;

// ---------------------------------------------------------------------------
// FP8 E4M3FN codec (bit-exact software reference).
// Format: 1 sign, 4 exponent (bias 7), 3 mantissa bits. No infinities; the
// only NaN encodings are 0x7f / 0xff. Largest finite magnitude is 448
// (S.1111.110). Subnormals are k * 2^-9, k = 1..7.
// Decoding never produces NaN/inf from finite codes. Encoding is RNE with
// saturation of overflowing finite magnitudes to +/-448; NaN inputs map to
// the positive quiet NaN code 0x7f.
// ---------------------------------------------------------------------------
inline constexpr float kFp8E4M3Max = 448.0f;
inline constexpr uint8_t kFp8E4M3Nan = 0x7fu;
[[nodiscard]] float fp8_e4m3_to_f32(uint8_t code) noexcept;
[[nodiscard]] uint8_t f32_to_fp8_e4m3(float f) noexcept;

// ---------------------------------------------------------------------------
// IEEE FP16 codec (bit-exact; used only for the Q8_0 block scale, which the
// GGUF stores as one FP16 per 32-code block).
// Decoding is exact for every code (subnormals, signed zero, inf, quiet NaN).
// Encoding is RNE; it exists for synthetic Q8_0 fixtures and tests -- the
// loader NEVER converts a scale back to FP16.
// ---------------------------------------------------------------------------
[[nodiscard]] float fp16_to_f32(uint16_t bits) noexcept;
[[nodiscard]] uint16_t f32_to_fp16(float f) noexcept;

// ---------------------------------------------------------------------------
// Q8_0 block geometry (GGUF Q8_0: 34 bytes per 32 weights).
// ---------------------------------------------------------------------------
inline constexpr uint32_t kQ8_0BlockSize = 32u;
inline constexpr uint64_t kQ8_0BlockBytes = 34u;// 2-byte FP16 scale + 32 int8

// Signed int8 code -> FP32 (exact; the same identity the device kernels use:
// ((code ^ 0x80) - 128) is the sign extension, done in integer arithmetic).
[[nodiscard]] inline float int8_to_f32(uint8_t code) noexcept {
    return static_cast<float>(static_cast<int8_t>(code));
}

// ---------------------------------------------------------------------------
// Storage-ABI packing helpers.
// ---------------------------------------------------------------------------
// Pack BF16 code units two per little-endian uint32_t word, low element
// first. An odd trailing element is zero-padded in the high half.
// packed_out.size() must be exactly ceil(codes.size() / 2).
[[nodiscard]] Status pack_bf16_pairs(luisa::span<const uint16_t> codes,
                                     luisa::span<uint32_t> packed_out) noexcept;
// Unpack; `count` is the number of valid BF16 elements (<= packed.size()*2).
// When count is odd the padding high half of the last word must be zero.
[[nodiscard]] Status unpack_bf16_pairs(luisa::span<const uint32_t> packed, uint64_t count,
                                       luisa::span<uint16_t> codes_out) noexcept;
// Pack FP8 E4M3 codes four per little-endian uint32_t word, low element
// first. Trailing elements of a partial word are zero-padded.
// packed_out.size() must be exactly ceil(codes.size() / 4).
[[nodiscard]] Status pack_fp8_quads(luisa::span<const uint8_t> codes,
                                    luisa::span<uint32_t> packed_out) noexcept;
[[nodiscard]] Status unpack_fp8_quads(luisa::span<const uint32_t> packed, uint64_t count,
                                      luisa::span<uint8_t> codes_out) noexcept;

// ---------------------------------------------------------------------------
// Weight storage classification (contract: dense F32 for sensitive tensors).
// Dense: every 1-D tensor (norms, ssm_a, ssm_dt.bias), the depthwise
// ssm_conv1d.weight, and any tensor already stored as F32 in the GGUF.
// Matrices stored as BF16 are packed BF16 or quantized to FP8 E4M3.
// ---------------------------------------------------------------------------
enum class WeightStorage : uint32_t {
    f32_dense = 0,  // Buffer<float>, one element per value
    bf16_packed,    // Buffer<uint32_t>, 2 BF16 codes per word
    fp8_packed,     // Buffer<uint32_t>, 4 E4M3 codes per word + FP32 scales
    q8_0_packed,    // Buffer<uint32_t>, 4 SIGNED int8 codes per word + FP32 scales
    w4_packed,      // Buffer<uint32_t>, 8 SIGNED int4 codes per word + FP32 scales
};
[[nodiscard]] luisa::string_view weight_storage_name(WeightStorage s) noexcept;

inline constexpr uint32_t kFp8GroupSize = 64u;

// Decide the storage for a GGUF tensor. `gguf_type` is the raw GGUF tensor
// type (0 = F32, 30 = BF16, 8 = Q8_0, ...), `dims` the ne-ordered dimensions.
// `want_fp8` selects the FP8 variant for eligible matrices. `want_q8_0`
// selects the NATIVE Q8_0 variant for Q8_0 matrices (byte copy of the block
// codes + exact FP16->FP32 scale widening); false keeps the historical
// explicit rejection (no silent reinterpretation, and Q8_0 is never FP8).
// `want_fp8` and `want_q8_0` are mutually exclusive.
[[nodiscard]] Result<WeightStorage> classify_weight_storage(
    luisa::string_view name, uint32_t gguf_type,
    luisa::span<const uint64_t> dims, bool want_fp8,
    bool want_q8_0 = false) noexcept;

// Number of FP32 scales for a matrix row of K elements: ceil(K / 64).
[[nodiscard]] inline uint64_t fp8_group_count(uint64_t k) noexcept {
    return (k + kFp8GroupSize - 1u) / kFp8GroupSize;
}

// Number of Q8_0 scales for a matrix row of K elements: K / 32 (whole blocks
// only; the caller validates K % 32 == 0, which the loader fails closed on).
[[nodiscard]] inline uint64_t q8_0_group_count(uint64_t k) noexcept {
    return k / kQ8_0BlockSize;
}

// ---------------------------------------------------------------------------
// Q8_0 storage-ABI packing (host; also used by the loader and by tests).
// ---------------------------------------------------------------------------
// Split a raw Q8_0 GGUF payload (blocks of 34 bytes: little-endian FP16 scale,
// then 32 signed int8 codes) into the device ABI: `words_out` (4 codes per
// uint32, LOW element first, one word per 4 consecutive elements) and
// `scales_out` (one FP32 exact widening of the block's FP16 scale per block).
// The int8 codes are COPIED -- no arithmetic, no rounding, no re-quantization.
// `element_count` must be a multiple of 32 and the payload exactly
// element_count/32 blocks.
[[nodiscard]] Status pack_q8_0_blocks(luisa::span<const std::byte> payload,
                                      uint64_t element_count,
                                      luisa::span<uint32_t> words_out,
                                      luisa::span<float> scales_out) noexcept;

// Inverse of the ABI for host references and tests: out[i] = int8(code) *
// scale[block]. `k` must be a multiple of 32 (whole blocks per row).
[[nodiscard]] Status dequantize_q8_0_matrix(luisa::span<const uint32_t> words,
                                            luisa::span<const float> scales,
                                            uint64_t k, uint64_t n,
                                            luisa::span<float> out) noexcept;

// ---------------------------------------------------------------------------
// W4A16 group-wise packed weights (opt_plan/05 Q2).
//
// ABI: Buffer<uint32_t> with EIGHT signed 4-bit two's-complement codes per
// little-endian word, LOW element first (element k of a row lives in nibble
// k % 8 of word k / 8). The code range is [-8, 7] (symmetric, NO zero point),
// so
//     dequantized weight = int4(code) * scale[row * ceil(K/group) + k / group]
// where the scale is one FP32 value per contiguous group of `group` K elements
// per output row (K % group == 0 is required, so groups are never ragged), and
// int4(code) is the exact integer sign extension ((code ^ 8) - 8).
//
// The scales are produced by a DETERMINISTIC host clipping search per group
// (W4ClipSearchConfig): the max-abs scale is multiplied by a sweep of clipping
// ratios in [min_ratio, max_ratio] and the ratio minimizing the group's
// dequantization MSE (computed in double against the stored FP32 scale) wins.
// That is the AWQ/GPTQ-style clipping trick -- plainly better than plain RTN --
// and it is fully reproducible: the same input bytes and config always produce
// byte-identical words and scales (no RNG, no FP contraction, ties keep the
// SMALLER ratio).
//
// Geometry is fail-closed: K > 0, K % 8 == 0 (whole words) and K % group == 0
// (whole groups), group in {32, 64, 128}. K % 8 == 0 and K % group == 0 are
// checked by validate_w4_geometry(); pack_w4_matrix also rejects non-finite
// inputs (NaN/inf) instead of writing codes for them.
//
// W4 is only ever produced from an F32/BF16 matrix (the offline packer); it is
// NEVER substituted for FP8/Q8_0/BF16 silently -- the storage kind is explicit
// everywhere (see the dsl-qwen weight-format enum).
// ---------------------------------------------------------------------------
inline constexpr uint32_t kW4CodesPerWord = 8u;
inline constexpr uint32_t kW4DefaultGroup = 128u;
inline constexpr uint32_t kW4CodeCount = 16u;// 4-bit code space
inline constexpr float kW4ScaleDenominator = 7.0f;

// int4 code (nibble, 0..15) -> FP32 value in [-8, 7], exact integer
// sign extension. Identical to the device kernels' ((code ^ 8) - 8).
[[nodiscard]] inline float w4_code_to_f32(uint32_t nibble) noexcept {
    return static_cast<float>(static_cast<int>((nibble & 0xFu) ^ 0x8u) - 8);
}

// Number of FP32 scales for a matrix row of K elements: K / group (whole
// groups only; the loader fails closed on K % group != 0).
[[nodiscard]] inline uint64_t w4_group_count(uint64_t k, uint32_t group) noexcept {
    return group == 0u ? 0u : k / group;
}

// Fail-closed W4 geometry: group in {32, 64, 128}, K > 0, K % 8 == 0 and
// K % group == 0. Never throws; the Status names the first violation.
[[nodiscard]] Status validate_w4_geometry(uint64_t k, uint32_t group) noexcept;

// Deterministic per-group clipping search configuration. `steps` is the size of
// the inclusive ratio grid; the canonical config is 31 steps over [0.85, 1.0]
// (step 0.005). `min_ratio <= max_ratio`, `min_ratio > 0`, `max_ratio <= 1`,
// `steps >= 2`.
struct W4ClipSearchConfig {
    uint32_t group{kW4DefaultGroup};
    float min_ratio{0.85f};
    float max_ratio{1.0f};
    uint32_t steps{31u};
    [[nodiscard]] Status validate() const noexcept;
    // Ratio of grid index i in [0, steps): min + i * (max - min) / (steps - 1).
    [[nodiscard]] double ratio_at(uint32_t i) const noexcept;
};

// Result statistics of one pack_w4_matrix call (host; reported, never used to
// change behavior). `mse`/`rtn_mse` are the mean-squared dequantization errors
// over every packed element, the first from the clipping search and the second
// from the plain max-abs/7 round-to-nearest reference (so the improvement is
// measurable, not claimed). `weighted_mse`/`rtn_weighted_mse` are the mean of the
// per-group IMPORTANCE-WEIGHTED objective over the groups of this call (see
// pack_w4_matrix_weighted); for an unweighted call they equal `mse`/`rtn_mse`.
struct W4PackStats {
    uint64_t groups{0};
    uint64_t clipped_groups{0};// groups whose best ratio is < 1.0
    double mse{0.0};
    double rtn_mse{0.0};
    double weighted_mse{0.0};
    double rtn_weighted_mse{0.0};
    double max_abs{0.0};
    float mean_ratio{1.0f};
};

// Quantize a whole [n, k] row-per-output FP32 matrix to the W4 ABI.
// `words_out.size()` must be exactly ceil(k*n/8) (k % 8 == 0 makes it exact),
// `scales_out.size()` exactly n * (k/group). Non-finite input, a mismatched
// span or geometry are recoverable failures; nothing is written on failure.
[[nodiscard]] Status pack_w4_matrix(luisa::span<const float> values, uint64_t k, uint64_t n,
                                    const W4ClipSearchConfig &cfg,
                                    luisa::span<uint32_t> words_out,
                                    luisa::span<float> scales_out,
                                    W4PackStats *stats = nullptr) noexcept;

// IMPORTANCE-AWARE group clipping (opt_plan/05 section 3 item 2).
//
// `importance` holds one measured per-input-channel statistic per K column,
//     m[j] = E[x[j]^2]   (the activation second moment written by
//                         qwen/w4_importance.h -- diag(H) under independent
//                         channels), the same m for every output row.
// The search minimizes, for every group g of every row,
//     E_g(s) = sum_{j in g} m[j] * (v[j] - q_j(s) * s)^2  /  sum_{j in g} m[j]
// i.e. the squared residual weighted by the measured activation energy of the
// channel and normalized by the group's total weight, so E_g is on the same scale
// as an unweighted group MSE (a constant weight vector cancels: m[j] == c
// reproduces pack_w4_matrix() byte for byte, which the host test asserts).
// Equivalently the residual is multiplied by sqrt(m[j]) before squaring, which is
// why the documentation calls it sqrt(importance)-weighted. Only the SCALE
// selection changes; codes stay nearest-int4 for the chosen scale, and ties still
// keep the smaller ratio.
//
// `mse`/`rtn_mse` keep their plain per-element meaning (comparable with an
// unweighted pack); `weighted_mse`/`rtn_weighted_mse` report the mean optimized
// objective over the groups of this call.
//
// Fail closed (invalid_argument, nothing written): `importance.size() != k`, a
// non-finite or negative value, or a group whose weight sum is not > 0 (those
// channels have no measured activation energy, so their objective is undefined).
[[nodiscard]] Status pack_w4_matrix_weighted(luisa::span<const float> values,
                                             luisa::span<const float> importance,
                                             uint64_t k, uint64_t n,
                                             const W4ClipSearchConfig &cfg,
                                             luisa::span<uint32_t> words_out,
                                             luisa::span<float> scales_out,
                                             W4PackStats *stats = nullptr) noexcept;

// Dequantize the W4 ABI back to FP32 (host reference / CPU oracle).
[[nodiscard]] Status dequantize_w4_matrix(luisa::span<const uint32_t> words,
                                          luisa::span<const float> scales,
                                          uint64_t k, uint64_t n, uint32_t group,
                                          luisa::span<float> out) noexcept;

// Decide the storage of a GGUF matrix in an OFFLINE W4 pack: dense F32 for
// every 1-D tensor (norms, ssm_a, ssm_dt.bias) and the depthwise convolution,
// w4_packed for every other matrix whose K satisfies validate_w4_geometry(). A
// tensor that cannot be packed is a recoverable failure (never a silent F32
// demotion of a matrix): the caller decides, the packer refuses.
[[nodiscard]] Result<WeightStorage> classify_w4_pack_storage(
    luisa::string_view name, luisa::span<const uint64_t> dims, uint32_t group) noexcept;

}// namespace luisa::qwen
