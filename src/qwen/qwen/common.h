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
// - Explicit bit-exact BF16 (RNE) and FP8 E4M3FN codecs plus the storage-ABI
//   packing helpers:
//     * BF16 tensors: two BF16 code units per little-endian uint32_t word,
//       LOW element first (raw copy of the GGUF BF16 bytes on LE hosts).
//     * FP8 E4M3 tensors: four E4M3 codes per little-endian uint32_t word,
//       LOW element first, FP32 scale per contiguous group of 64 K elements
//       per output row, scale = max_abs / 448.
//   FP8 here means E4M3FN (finite, max 448, NaN code 0x7f/0xff) -- never Q8_0.
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
};
[[nodiscard]] luisa::string_view weight_storage_name(WeightStorage s) noexcept;

inline constexpr uint32_t kFp8GroupSize = 64u;

// Decide the storage for a GGUF tensor. `gguf_type` is the raw GGUF tensor
// type (0 = F32, 30 = BF16, 8 = Q8_0, ...), `dims` the ne-ordered dimensions.
// `want_fp8` selects the FP8 variant for eligible matrices. Q8_0 matrices are
// rejected here with `unsupported` (no silent reinterpretation); callers must
// explicitly dequantize first. Non-float quant types are always rejected.
[[nodiscard]] Result<WeightStorage> classify_weight_storage(
    luisa::string_view name, uint32_t gguf_type,
    luisa::span<const uint64_t> dims, bool want_fp8) noexcept;

// Number of FP32 scales for a matrix row of K elements: ceil(K / 64).
[[nodiscard]] inline uint64_t fp8_group_count(uint64_t k) noexcept {
    return (k + kFp8GroupSize - 1u) / kFp8GroupSize;
}

}// namespace luisa::qwen
