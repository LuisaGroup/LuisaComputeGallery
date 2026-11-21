#pragma once
// Qwen3.5-4B full-attention KV cache -- DTYPE + LAYOUT contract.
//
// ONE header owns the element geometry of a K/V region so the allocator, the
// engine, the attention kernels and the tests can never disagree about it
// (opt_plan/05-quantization.md section 4 item Q3, 03-decode-latency.md section
// 4.3, 01-kv-cache-pages.md section 2.7).
//
// Host-only: no DSL types, no device includes. The DSL mirrors of these two
// functions live in qwen/dsl_ops.cpp (`qwen_kv_code_index` / `qwen_kv_load`) and
// are checked against this header by the device differential tests.
//
// Two independent axes:
//
//  dtype f32  -> one IEEE FP32 value per 32-bit word (4 B per value).
// bf16 -> two BF16 codes per 32-bit word (2 B per value), LOW code
// first (element 2k in bits 0..15, element 2k+1 in bits
// 16..31) -- the SAME word ABI as the BF16 weight tensors
// (qwen/common.h) so one decode helper serves both.
// The store side rounds FP32 -> BF16 with round-to-nearest-even
// (qwen::f32_to_bf16), never truncation and never a stochastic mode.
// fp8_e4m3 -> four E4M3FN codes per 32-bit word (1 B per value), LOW
// code first (element 4k in bits 0..7, 4k+3 in bits 24..31).
// The codec is BIT-IDENTICAL to the weight FP8 path
// (qwen::f32_to_fp8_e4m3 / qwen::fp8_e4m3_to_f32): E4M3FN,
// no infinities, largest finite magnitude 448, RNE with saturation
// of overflowing finite magnitudes, NaN stored as the NaN code.
// FP8 is a SCALED format: every (kv_head, position_block=64) pair
// owns one FP32 scale and the stored value is code * scale
// (see `scale_index`/`qwen_kv_block_scale` below).
// Activations, accumulators and softmax stay FP32 on every path: only
// the STORED K/V codes are narrow.
//
//  layout  position_major -> legacy [position][kv_head][channel]
//                           index = pos * (kv_heads * head_dim) + head * head_dim + channel
//                           (contiguous per token; the FP32 reference/A-B path)
//          blocked64      -> [kv_head][position_block][position_in_block][channel]
//                           index = ((head * position_blocks + pos / 64) * 64
//                                    + pos % 64) * head_dim + channel
//                           (block-major: a 64-position block of one head is one
//                           contiguous run of 64 * head_dim values, which is what
//                           makes a tile load contiguous in both channel and
//                           position -- opt_plan/03 section 4.3)
//
// Every index below is a CODE index inside ONE region (one layer's K or one
// layer's V), 0-based. A region is contiguous in the sparse virtual buffer; only
// this arithmetic decides where a value lives, so the sparse page model, the
// view-stability contract and the one-heap-per-page invariant are untouched.
//
// The native pair per dtype (qwen_kv_native_layout) is f32 -> position_major and
// bf16/fp8_e4m3 -> blocked64. Both cross combinations are legal for the UNSCALED
// dtypes and exist so the dtype and the layout can be measured separately; no path
// silently substitutes one for the other (the engine reports what it built). The
// SCALED dtype (fp8_e4m3) is only legal with blocked64, because its scale is
// indexed by position block: qwen_kv_layout_desc() fails closed otherwise.
//
// SCALE AXIS (fp8_e4m3 only). One FP32 scale per (kv_head, position_block): a
// region's scales are a DENSE [kv_heads][position_blocks] array, owned by the
// allocator outside the sparse buffer (so a scale read can never be unmapped).
//   scale index = head * position_blocks + position / 64
// Validity rule (the online-append contract the device kernels implement and
// `qwen_kv_append_reference` below defines):
//   * for every VALID position of a block, scale == max_abs(block) / 448, where
//     max_abs is the RUNNING maximum of |v| over every value ever appended to that
//     block in every channel (i.e. max_abs/448 over the block's valid values).
//     "Running" matters: with no FP32 shadow copy an append cannot re-derive the
//     old maximum from the stored codes, so the scale itself carries it and is
//     only ever RAISED (a block whose new values stay below 448*scale keeps its
//     scale and its bytes);
//   * a value is stored as round_e4m3(value / scale) and read back as
//     code * scale, so old codes are NEVER left paired with a new scale: growing
//     the block maximum rewrites the whole valid prefix of the block;
//   * a block that has never been written has scale 0 and no valid position
//     (a zero scale can never be read for a valid position, and an all-zero block
//     is representable with scale 0 since every code is then 0 too);
// * a non-finite source value contributes 448 to the block maximum instead of
//   infinity. That guarantees a positive scale even for a block containing only
//   inf/NaN (inf then saturates to +/-448*scale and NaN keeps its NaN code), so
//   invalid model data remains visible instead of becoming an innocent zero.

#include <cstdint>

#include <luisa/core/stl/vector.h>

#include "qwen/common.h"

namespace luisa::qwen {

enum class QwenKvDType : uint32_t {
    f32 = 0u,      // legacy: dense FP32 codes, one per 32-bit word
    bf16 = 1u,     // two RNE BF16 codes per 32-bit word, low code first
    fp8_e4m3 = 2u, // four E4M3FN codes per 32-bit word, low code first, SCALED
};

enum class QwenKvLayout : uint32_t {
    position_major = 0u, // legacy [position][kv_head][channel]
    blocked64 = 1u,      // [kv_head][position_block=64][position%64][channel]
};

// Position-block extent of the blocked layout. Compile-time because it is baked
// into every attention kernel's index arithmetic and into the allocator's
// residency set (a "position block" is the unit that must stay contiguous).
inline constexpr uint32_t qwen_kv_position_block = 64u;

// Bytes of ONE stored K/V value.
[[nodiscard]] constexpr uint32_t qwen_kv_element_bytes(QwenKvDType dtype) noexcept {
    switch (dtype) {
        case QwenKvDType::bf16: return 2u;
        case QwenKvDType::fp8_e4m3: return 1u;
        case QwenKvDType::f32:
        default: return 4u;
    }
}

// Stored values per 32-bit word (1 FP32, 2 BF16, 4 FP8 E4M3).
[[nodiscard]] constexpr uint32_t qwen_kv_elements_per_word(QwenKvDType dtype) noexcept {
    switch (dtype) {
        case QwenKvDType::bf16: return 2u;
        case QwenKvDType::fp8_e4m3: return 4u;
        case QwenKvDType::f32:
        default: return 1u;
    }
}

// True when the dtype needs one FP32 scale per (kv_head, position_block). Only
// fp8_e4m3 is scaled; f32 and bf16 are self-contained codes.
[[nodiscard]] constexpr bool qwen_kv_scaled(QwenKvDType dtype) noexcept {
    return dtype == QwenKvDType::fp8_e4m3;
}

// The layout a dtype is validated in: FP32 keeps the legacy position-major
// layout (that is the A/B reference the whole plan compares against), BF16 and
// FP8 E4M3 use the blocked layout (FP8 because its scale is block-indexed).
[[nodiscard]] constexpr QwenKvLayout qwen_kv_native_layout(QwenKvDType dtype) noexcept {
    return dtype == QwenKvDType::f32 ? QwenKvLayout::position_major
                                     : QwenKvLayout::blocked64;
}

[[nodiscard]] constexpr luisa::string_view qwen_kv_dtype_name(QwenKvDType dtype) noexcept {
    switch (dtype) {
        case QwenKvDType::bf16: return luisa::string_view{"bf16"};
        case QwenKvDType::fp8_e4m3: return luisa::string_view{"fp8_e4m3"};
        case QwenKvDType::f32:
        default: return luisa::string_view{"f32"};
    }
}

[[nodiscard]] constexpr luisa::string_view qwen_kv_layout_name(QwenKvLayout layout) noexcept {
    return layout == QwenKvLayout::blocked64 ? luisa::string_view{"blocked64"}
                                             : luisa::string_view{"position_major"};
}

// "--kv-dtype f32|bf16|fp8". `fp8`/`e4m3`/`fp8_e4m3`/`e4m3fn` all select
// fp8_e4m3 (the log always prints the canonical "fp8_e4m3"). Returns false on an
// unknown name (never a default).
[[nodiscard]] inline bool qwen_kv_dtype_from_name(luisa::string_view name,
                                                  QwenKvDType &out) noexcept {
    if (name == "f32" || name == "fp32" || name == "float32") {
        out = QwenKvDType::f32;
        return true;
    }
    if (name == "bf16" || name == "bfloat16") {
        out = QwenKvDType::bf16;
        return true;
    }
    if (name == "fp8" || name == "e4m3" || name == "fp8_e4m3" || name == "e4m3fn" ||
        name == "fp8e4m3") {
        out = QwenKvDType::fp8_e4m3;
        return true;
    }
    return false;
}

[[nodiscard]] inline bool qwen_kv_layout_from_name(luisa::string_view name,
                                                   QwenKvLayout &out) noexcept {
    if (name == "position_major" || name == "legacy") {
        out = QwenKvLayout::position_major;
        return true;
    }
    if (name == "blocked64" || name == "blocked") {
        out = QwenKvLayout::blocked64;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Per-region geometry, in 32-bit words and in codes.
// ---------------------------------------------------------------------------
struct QwenKvLayoutDesc {
    QwenKvDType dtype{QwenKvDType::f32};
    QwenKvLayout layout{QwenKvLayout::position_major};
    uint32_t max_context{0};   // positions addressable in the region
    uint32_t kv_heads{0};      // kv_heads * head_dim == values_per_token
    uint32_t head_dim{0};
    // blocked64 only: ceil(max_context / 64) position blocks per head. The last
    // block is padded up to a whole 64-position block, which is the ONLY padding
    // this layout introduces (it is inside the region and therefore addressable,
    // unlike the page padding the allocator adds on top).
    uint32_t position_blocks{0};
    // blocked64 only: codes per head = position_blocks * 64 * head_dim.
    uint64_t head_stride_codes{0};
    // Codes actually addressable in the region, padding included.
    uint64_t region_codes{0};
    // The same in words (the allocator's and the buffer views' unit).
    uint64_t region_words{0};
    // fp8_e4m3 only: one FP32 scale per (kv_head, position_block), a DENSE array
    // [kv_heads][position_blocks] laid out head-major. 0 for the unscaled dtypes.
    uint32_t scale_count{0};

    [[nodiscard]] constexpr uint32_t elements_per_word() const noexcept {
        return qwen_kv_elements_per_word(dtype);
    }
    // Codes per token with the LEGACY layout (position_major payload), i.e.
    // kv_heads * head_dim. This is also values_per_token()/attn_kv_n().
    [[nodiscard]] constexpr uint64_t legacy_codes_per_token() const noexcept {
        return static_cast<uint64_t>(kv_heads) * head_dim;
    }
    // Payload codes of one region WITHOUT the 64-position block padding. For
    // position_major it equals region_codes; for blocked64 the difference is the
    // tail padding of every head (at most 63 * head_dim codes per head).
    [[nodiscard]] constexpr uint64_t payload_codes() const noexcept {
        return static_cast<uint64_t>(max_context) * legacy_codes_per_token();
    }
    [[nodiscard]] constexpr uint64_t padding_codes() const noexcept {
        return region_codes - payload_codes();
    }
    // Bytes of one region without/with the block padding.
    [[nodiscard]] constexpr uint64_t payload_bytes() const noexcept {
        return payload_codes() * qwen_kv_element_bytes(dtype);
    }
    [[nodiscard]] constexpr uint64_t region_bytes() const noexcept {
        return region_codes * qwen_kv_element_bytes(dtype);
    }
    // Bytes stored per context position (all heads, all channels) -- the number
    // the capacity model and the benchmark documentation quote.
    [[nodiscard]] constexpr uint64_t bytes_per_token() const noexcept {
        return legacy_codes_per_token() * qwen_kv_element_bytes(dtype);
    }
    // Codes of one 64-position block of one head (blocked64 only; 0 otherwise).
    [[nodiscard]] constexpr uint64_t block_codes() const noexcept {
        return static_cast<uint64_t>(qwen_kv_position_block) * head_dim;
    }
    [[nodiscard]] constexpr bool scaled() const noexcept { return qwen_kv_scaled(dtype); }
    // Bytes of the dense scale array of ONE region (0 for the unscaled dtypes).
    [[nodiscard]] constexpr uint64_t scale_bytes() const noexcept {
        return static_cast<uint64_t>(scale_count) * sizeof(float);
    }
    // Index of the FP32 scale that governs (head, position): head-major over the
    // position blocks of that head. The caller guarantees the range; 0 for the
    // unscaled dtypes (the value is never used then).
    [[nodiscard]] constexpr uint64_t scale_index(uint32_t head, uint64_t position) const noexcept {
        if (scale_count == 0u) { return 0u; }
        return static_cast<uint64_t>(head) * position_blocks +
               position / qwen_kv_position_block;
    }
    // Blocks a `tokens`-position prefix of one head occupies (>= 1 when tokens > 0).
    [[nodiscard]] constexpr uint64_t head_blocks(uint32_t tokens) const noexcept {
        return (static_cast<uint64_t>(tokens) + qwen_kv_position_block - 1u) /
               qwen_kv_position_block;
    }

    // Code index of (head, position, channel) inside the region. The caller
    // guarantees head < kv_heads, position < max_context, channel < head_dim.
    [[nodiscard]] constexpr uint64_t code_index(uint32_t head, uint64_t position,
                                                uint32_t channel) const noexcept {
        if (layout == QwenKvLayout::position_major) {
            return position * legacy_codes_per_token() +
                   static_cast<uint64_t>(head) * head_dim + channel;
        }
        const uint64_t block = position / qwen_kv_position_block;
        const uint64_t in_block = position % qwen_kv_position_block;
        return static_cast<uint64_t>(head) * head_stride_codes +
               block * block_codes() + in_block * head_dim + channel;
    }
    // Word index and in-word bit shift of that code (shift 0 for f32, 0 or 16
    // for bf16). Every base term is an even number of codes for BF16 because
    // head_dim is required even, so a code pair never crosses a row.
    [[nodiscard]] constexpr uint64_t word_index(uint64_t code) const noexcept {
        return code / elements_per_word();
    }
    // Bit offset of a code inside its 32-bit word: 8 bits per stored byte, so 0/16
    // for bf16, 0/8/16/24 for fp8_e4m3 and 0 for f32. (Deriving it from
    // element_bytes rather than hard-coding 16 is what keeps the fp8 quad packed
    // the way common.h's pack_fp8_quads packs it.)
    [[nodiscard]] constexpr uint32_t word_shift(uint64_t code) const noexcept {
        return static_cast<uint32_t>((code % elements_per_word()) *
                                    (8u * qwen_kv_element_bytes(dtype)));
    }
    // First code of one head, and the CONTIGUOUS run of codes that covers
    // `tokens` positions of that head. With position_major the layout is not
    // head-major, so `head_run_codes` is only meaningful for blocked64; callers
    // use codes_for_positions() for the position-major residency extent.
    [[nodiscard]] constexpr uint64_t head_base_codes(uint32_t head) const noexcept {
        return layout == QwenKvLayout::position_major
                   ? static_cast<uint64_t>(head) * head_dim
                   : static_cast<uint64_t>(head) * head_stride_codes;
    }
    [[nodiscard]] constexpr uint64_t head_run_codes(uint32_t head, uint32_t tokens) const noexcept {
        if (layout == QwenKvLayout::position_major) { return 0u; }
        const uint64_t blocks = (tokens + qwen_kv_position_block - 1u) / qwen_kv_position_block;
        const uint64_t capped = blocks < position_blocks ? blocks : position_blocks;
        return capped * block_codes();
    }
    // Codes touched in the region by `tokens` context positions with the
    // position_major layout: a contiguous prefix (channels of every head of each
    // token interleave, so partial tokens are not expressible per head).
    [[nodiscard]] constexpr uint64_t prefix_codes(uint32_t tokens) const noexcept {
        return static_cast<uint64_t>(tokens) * legacy_codes_per_token();
    }
    // Positions a single page covers for ONE head in the blocked layout, i.e. how
    // coarse residency is in token terms. 0 when the page is smaller than one
    // block (the allocator's residency math does not depend on this; it only
    // feeds the diagnostic line).
    [[nodiscard]] constexpr uint64_t positions_per_page_per_head(uint64_t page_words) const noexcept {
        const uint64_t code_bytes = qwen_kv_element_bytes(dtype);
        const uint64_t row_bytes = static_cast<uint64_t>(head_dim) * code_bytes;
        return row_bytes == 0u ? 0u : (page_words * 4u) / row_bytes;
    }
};

// Validate + fill `out`. Fail-closed on every geometry the kernels could not
// index, with a message naming the offending field:
//  * max_context/kv_heads/head_dim nonzero and head_dim even for bf16 (a code
//    pair must not straddle a position row);
//  * blocked64 requires an even head_dim for the same reason (the layout is
//    used with bf16 in practice; f32 blocked64 is exact for any head_dim);
//  * every product checked (uint64 overflow -> out_of_memory).
[[nodiscard]] Status qwen_kv_layout_desc(QwenKvDType dtype, QwenKvLayout layout,
                                         uint32_t max_context, uint32_t kv_heads,
                                         uint32_t head_dim, QwenKvLayoutDesc &out) noexcept;

// Convenience: same as above with layout = qwen_kv_native_layout(dtype).
[[nodiscard]] Status qwen_kv_native_desc(QwenKvDType dtype, uint32_t max_context,
                                         uint32_t kv_heads, uint32_t head_dim,
                                         QwenKvLayoutDesc &out) noexcept;

// ---------------------------------------------------------------------------
// Host codec helpers (BF16 word ABI, identical to the weight tensors).
// ---------------------------------------------------------------------------
// Code of one value (RNE). FP32 values beyond BF16's range become +/-inf and
// NaN stays a quiet BF16 NaN -- qwen::f32_to_bf16 is the single implementation.
[[nodiscard]] inline uint16_t qwen_kv_code_from_f32(float v) noexcept {
    return f32_to_bf16(v);
}
[[nodiscard]] inline float qwen_kv_f32_from_code(uint16_t code) noexcept {
    return bf16_to_f32(code);
}
// FP8 E4M3FN code of one value (RNE, saturating, NaN -> the NaN code). This is
// the WEIGHT codec (qwen/common.h), not a KV-specific one: the same 256 E4M3
// codes, the same 448 saturation point, the same round-to-nearest-even, so a
// test that validates the weight path validates the KV path too.
[[nodiscard]] inline uint8_t qwen_kv_fp8_code_from_f32(float v) noexcept {
    return f32_to_fp8_e4m3(v);
}
[[nodiscard]] inline float qwen_kv_fp8_code_to_f32(uint8_t code) noexcept {
    return fp8_e4m3_to_f32(code);
}
// FP8 code of a value that is stored with `scale`: round_e4m3(value / scale).
// A non-positive scale means "the block has no scale yet", which is only legal
// while every stored value is exactly zero (see the scale rule above); the
// division is then replaced by 0 so no 0/0 NaN can enter the cache.
[[nodiscard]] inline uint8_t qwen_kv_fp8_code_from_scaled(float v, float scale) noexcept {
    return f32_to_fp8_e4m3(scale > 0.0f ? v / scale : 0.0f);
}
// Scale of a block whose maximum magnitude is `max_abs`: max_abs / 448, and 0
// for an all-zero (or never-written) block. `max_abs` must be finite and
// non-negative: the append contract keeps non-finite source values out of the
// block maximum (see the header comment).
// 1/448 as ONE rounded FP32 constant. The scale is computed as max_abs *
// kQwenKvInvFp8Max on the host AND on the device, so the two are bit-identical:
// a division can be lowered to a reciprocal multiply by a fast-math build, which
// would move the last bit of every scale (and therefore some codes) between the
// engine and the CPU oracle. The error against max_abs/448 is <= 1 ulp and the
// encoder saturates the single value that could exceed 448, so nothing clips.
inline constexpr float kQwenKvInvFp8Max = 1.0f / kFp8E4M3Max;
[[nodiscard]] inline float qwen_kv_block_scale(float max_abs) noexcept {
    return max_abs > 0.0f ? max_abs * kQwenKvInvFp8Max : 0.0f;
}
// Largest magnitude the block with `scale` can represent without saturating
// (448 * scale). Used by the append path as the "does the block maximum grow"
// test: a new value <= this bound needs no rescale of the earlier codes.
[[nodiscard]] inline float qwen_kv_block_limit(float scale) noexcept {
    return scale > 0.0f ? scale * kFp8E4M3Max : 0.0f;
}
// True for a finite value. The append maximum maps false (non-finite) to one
// full-scale unit (448) instead, so invalid data cannot zero its block scale.
[[nodiscard]] inline bool qwen_kv_counts_for_max(float v) noexcept {
    return v == v && v > -3.402823466e38f && v < 3.402823466e38f;
}
// Word pack/unpack: LOW code is element 2k, HIGH code is element 2k+1.
[[nodiscard]] inline uint32_t qwen_kv_pack_codes(uint16_t lo, uint16_t hi) noexcept {
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16u);
}
[[nodiscard]] inline uint16_t qwen_kv_unpack_code(uint32_t word, uint32_t bit_shift) noexcept {
    return static_cast<uint16_t>((word >> bit_shift) & 0xFFFFu);
}
// Reinterpret one buffer WORD of a region as the stored FP32 value (f32 dtype).
// A BF16 word must never be read this way: qwen_kv_read_word is the dtype-aware
// accessor and the tests use it for both dtypes.
[[nodiscard]] inline float qwen_kv_word_as_f32(uint32_t word) noexcept {
    float f;
    std::memcpy(&f, &word, sizeof(f));
    return f;
}
[[nodiscard]] inline uint32_t qwen_kv_f32_as_word(float v) noexcept {
    uint32_t w;
    std::memcpy(&w, &v, sizeof(w));
    return w;
}
// Dtype-aware word accessor: the stored value at code `code` inside a region
// whose 32-bit words are `words`. `scale` is the (head, position-block) FP32
// scale of that code and is only applied for fp8_e4m3 (the unscaled dtypes are
// self-contained codes, so the default 1 is exact for them). Every caller that
// reads an fp8 region must pass the scale of the code's own block -- that is the
// invariant the whole append design exists to keep.
[[nodiscard]] inline float qwen_kv_read_word(const QwenKvLayoutDesc &desc,
                                             luisa::span<const uint32_t> words,
                                             uint64_t code, float scale = 1.0f) noexcept {
    const uint64_t w = desc.word_index(code);
    if (desc.dtype == QwenKvDType::f32) { return qwen_kv_word_as_f32(words[w]); }
    if (desc.dtype == QwenKvDType::fp8_e4m3) {
        const auto byte = static_cast<uint8_t>((words[w] >> desc.word_shift(code)) & 0xFFu);
        return qwen_kv_fp8_code_to_f32(byte) * scale;
    }
    return qwen_kv_f32_from_code(qwen_kv_unpack_code(words[w], desc.word_shift(code)));
}

// Dequantize one region (`tokens` positions, all heads, all channels) into a
// DENSE FP32 [position][kv_head][channel] image -- exactly the legacy layout the
// qwen::ref_qwen_attn_* CPU oracles consume, so a BF16/FP8 cache can be compared
// against the same oracle the FP32 path is compared against. For f32 + the
// position_major layout this is a byte copy. `dense.size()` must be at least
// tokens * kv_heads * head_dim; the tail (codes a BF16/FP8 block pads) is ignored.
[[nodiscard]] Status qwen_kv_region_to_dense(const QwenKvLayoutDesc &desc,
                                             luisa::span<const uint32_t> words,
                                             uint32_t tokens,
                                             luisa::span<float> dense) noexcept;

// The same, applying the per-(head, block) scale of a SCALED region. `scales`
// must hold desc.scale_count values for fp8_e4m3 and may be empty for the
// unscaled dtypes (the 4-argument overload above forwards to this one with an
// empty span, i.e. it refuses an fp8 region rather than reading code bytes as
// values).
[[nodiscard]] Status qwen_kv_region_to_dense(const QwenKvLayoutDesc &desc,
                                             luisa::span<const uint32_t> words,
                                             luisa::span<const float> scales,
                                             uint32_t tokens,
                                             luisa::span<float> dense) noexcept;

// Dense FP32 [position][kv_head][channel] -> region words in the given layout
// (the inverse of the above; used by the tests to build a cache from a dense
// fixture without duplicating the index arithmetic). For a SCALED dtype this is
// the ONLINE APPEND contract at pos0 == 0 covering `tokens` positions: every
// block scale is max_abs/448 over that block's valid values and the codes are
// encoded against it. An unscaled dtype ignores `scales`.
[[nodiscard]] Status qwen_kv_region_from_dense(const QwenKvLayoutDesc &desc,
                                               luisa::span<const float> dense,
                                               uint32_t tokens,
                                               luisa::span<uint32_t> words) noexcept;
[[nodiscard]] Status qwen_kv_region_from_dense(const QwenKvLayoutDesc &desc,
                                               luisa::span<float> scales,
                                               luisa::span<const float> dense,
                                               uint32_t tokens,
                                               luisa::span<uint32_t> words) noexcept;

// ---------------------------------------------------------------------------
// Online append (the CPU oracle the device append kernel is compared against,
// and the definition of the scale rule above).
//
// `src` holds `tokens` rows of `desc.legacy_codes_per_token()` values at
// absolute positions [pos0, pos0+tokens). For each (kv_head, 64-position block)
// the append touches:
//   max_abs = max |v| over the NEW rows only (non-finite values excluded),
//   scale'  = max(scale, max_abs/448)          (the scale carries the old maximum),
// and the whole valid prefix of the block is rewritten against scale' when it grew
// (earlier codes re-encoded from their OLD scale, so none is ever left paired with
// the new one). When it does not grow only the new rows are written and every
// earlier byte is untouched. This is exactly what
// make_qwen_kv_append_fp8_kernel() does on the device -- the two are compared
// word-for-word by the device suite.
//
// Both the f32 and the bf16 dtypes are supported (the scale array is then
// unused and may be empty); for fp8_e4m3 `scales` must hold desc.scale_count
// values and is read/written. `words` must hold desc.region_words entries.
[[nodiscard]] Status qwen_kv_append_reference(const QwenKvLayoutDesc &desc,
                                              luisa::span<const float> src,
                                              uint32_t pos0, uint32_t tokens,
                                              luisa::span<float> scales,
                                              luisa::span<uint32_t> words) noexcept;

}// namespace luisa::qwen
