// Qwen sparse KV-cache allocator device regression test (tests/integration/qwen).
//
// Usage: test_qwen_kv_cache <backend>       e.g. test_qwen_kv_cache cuda
// The backend is EXPLICIT argv[1]; there is no default and no silent skip.
// Requires no checkpoint: the allocator is driven with synthetic data only.
// Exits nonzero on any failure. The ONLY allowed skip is create() returning
// StatusCode::unsupported because the backend has no sparse-buffer support;
// every other create/ensure/reset failure is a failure.
//
// Vocabulary: tile = backend mapping granularity (never changed, 2 MiB on CUDA
// and 64 KiB on DX/VK); page = ALLOCATION unit = a whole number of tiles and at
// least 2 MiB; one heap per page, mapped by exactly one map_tile call. Every
// token count and every expected residency below is derived from the PROBED tile
// size -- nothing is assumed about a specific backend.
//
// Coverage:
//  - probe-stage allocator: tile-size discovery, on-demand commit, and
//    destruction while a dispatched write may still be in flight.
//  - main allocator: zero residency after create, view sizes/distinct
//    page-aligned region bases, idempotent ensure_capacity, growth across a
//    real PAGE boundary (the page size and the token count that crosses one are
//    derived from the probed tile size, never assumed), exact-pattern writes
//    through sparse k_view/v_view on two layer pairs, data preservation across
//    growth, reset after dispatched work (bookkeeping back to zero, buffer
//    reusable), and deferred teardown after an in-flight dispatch.
//  - page-model invariants: page_size_bytes() >= 2 MiB and an exact multiple of
//    tile_size_bytes(), tiles_per_page() == page/tile, pages_per_region() ==
//    ceil(logical bytes / page bytes), mapped_heap_count() ==
//    committed_pages_per_region() * region_count, committed_bytes() ==
//    mapped_heap_count() * page_size_bytes().
//  - NO-ALIAS PROBE (the important one): every (region, page) gets a
//    region-and-page-distinct FP32 pattern written into the FIRST and LAST tile
//    of that page and read back. Two pages ever mapped from one heap base (the
//    DX RangeStart trap) collide on the first element compared.
//  - one contiguous write straddling a page boundary inside one region.
//  - growth that crosses a page boundary preserves every earlier byte.
//  - reset() then re-ensure_capacity() to the same and to a larger token count
//    reuses the same virtual addresses (views stay valid) and remaps the right
//    page count.
//  - envelope guards fail closed: tokens > max_context -> invalid_argument,
//    total-tile overflow -> out_of_memory, unusable page hint ->
//    invalid_argument.
//  - kv_page_bytes == tile_bytes (legacy one-tile-page configuration) is
//    byte-identical to the default paged configuration for the same tokens.
//  - HOST contract stage (no device): the KV dtype/layout contract of
//    qwen/kv_cache_layout.h -- BF16 RNE ties (both parities), subnormals, negative
//    codes, NaN/inf, the blocked region arithmetic, the 64-position padding, exact
//    address injectivity and bounds over a ragged 19250-position context, and the
//    page/token coverage numbers per dtype/layout.
//  - BF16 + blocked64 device stage: dtype/layout accessors, region/payload/logical
//    byte geometry, the per-head PAGE-UNION residency at 1/64/65/5000/17000/19250
//    positions (a linear-prefix mapping fails here), per-head no-alias markers,
//    growth data preservation, reset/remap at the same addresses, and the FP32 A/B
//    at the same geometry (twice the payload, one page per region for a single
//    token instead of the BF16 head union).
#include "qwen_device_test_utils.h"

#include "qwen/kv_cache_allocator.h"

#include <luisa/core/stl/memory.h>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>// std::numeric_limits (bf16 specials in the host contract stage)

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::qwen;
using namespace qwen_test;

namespace {

// Exact-pattern writer: out[i] = scale * i. Powers-of-two scales keep every
// product of the small indices used here bit-exact in FP32, so device and
// host patterns compare exactly.
using WriteKvKernel = Kernel1D<void(Buffer<float>, Buffer<float>,
                                    uint32_t, float, float)>;

WriteKvKernel make_write_kv_kernel() noexcept {
    return [](BufferFloat k, BufferFloat v, Var<uint> count,
              Var<float> k_scale, Var<float> v_scale) noexcept {
        set_block_size(256u, 1u, 1u);
        const auto i = dispatch_x();
        $if (i < count) {
            k.write(i, k_scale * cast<float>(i));
            v.write(i, v_scale * cast<float>(i));
        };
    };
}

// Window writer for the no-alias probe: out[base + i] = tag + i. One dispatch
// per (region, page, page end) gives every probed 32-element window a unique,
// host-reproducible tag, so a page that aliases another page is detected on the
// first compared element.
using WriteWindowKernel = Kernel1D<void(Buffer<float>, uint32_t, uint32_t, float)>;

WriteWindowKernel make_write_window_kernel() noexcept {
    return [](BufferFloat buf, Var<uint> base, Var<uint> count, Var<float> tag) noexcept {
        set_block_size(64u, 1u, 1u);
        const auto i = dispatch_x();
        $if (i < count) { buf.write(base + i, tag + cast<float>(i)); };
    };
}

[[nodiscard]] luisa::vector<float> pattern(uint32_t count, float scale) {
    luisa::vector<float> host(count);
    for (uint32_t i = 0u; i < count; ++i) {
        host[i] = scale * static_cast<float>(i);
    }
    return host;
}

[[nodiscard]] luisa::vector<float> window_pattern(float tag, uint32_t count) {
    luisa::vector<float> host(count);
    for (uint32_t i = 0u; i < count; ++i) {
        host[i] = tag + static_cast<float>(i);
    }
    return host;
}

// Spec-level ceil-div, used only to state the EXPECTED page geometry
// independently of the allocator under test.
[[nodiscard]] uint64_t ceil_div(uint64_t a, uint64_t b) noexcept {
    return a / b + (a % b != 0u ? 1u : 0u);
}

template<typename Shader>
void dispatch_write(Shader &shader, Stream &stream, BufferView<float> k, BufferView<float> v,
                    uint32_t count, float k_scale, float v_scale) {
    stream << shader(k, v, count, k_scale, v_scale).dispatch(count);
}

template<typename Shader>
void dispatch_window(Shader &shader, Stream &stream, BufferView<float> view,
                     uint32_t base, uint32_t count, float tag) {
    stream << shader(view, base, count, tag).dispatch(count);
}

[[nodiscard]] luisa::vector<float> read_back(Stream &stream, BufferView<float> view) {
    luisa::vector<float> host(view.size());
    stream << view.copy_to(luisa::span{host}) << synchronize();
    return host;
}

}// namespace

int main(int argc, char **argv) {
    unbuffered_output();
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <backend>\n", argv[0]);
        return 2;
    }
    // ------------------------------------------------------------------
    // Stage 0 (HOST, no device): the KV dtype/layout CONTRACT itself
    // (opt_plan/05-quantization.md Q3). These checks need no backend, so
    // they run before any device exists and pin the arithmetic every
    // kernel and the allocator share (qwen/kv_cache_layout.h).
    // ------------------------------------------------------------------
    {
        // BF16 codec: RNE, ties to even, subnormals, negatives, NaN/inf.
        check(f32_to_bf16(1.00390625f) == 0x3F80u && f32_to_bf16(1.01171875f) == 0x3F82u,
              "kv contract: bf16 ties round to even (both parities)");
        check(f32_to_bf16(-1.0f) == 0xBF80u && bf16_to_f32(0xBF80u) == -1.0f,
              "kv contract: negative codes keep their sign bit");
        const float tiny = std::ldexp(1.0f, -131);
        check(f32_to_bf16(tiny) == 0x0004u && f32_to_bf16(-tiny) == 0x8004u &&
                  bf16_to_f32(0x0004u) == tiny,
              "kv contract: bf16 subnormal codes round and widen exactly");
        check(std::isnan(bf16_to_f32(f32_to_bf16(std::numeric_limits<float>::quiet_NaN()))) &&
                  std::isinf(bf16_to_f32(f32_to_bf16(-std::numeric_limits<float>::infinity()))),
              "kv contract: NaN stays a quiet NaN, infinity is preserved");
        check(std::signbit(bf16_to_f32(f32_to_bf16(-0.0f))),
              "kv contract: -0.0 keeps its sign");

        // Ragged 64-position blocked geometry (max_context is NOT a multiple
        // of 64, so the last block of every head is padded).
        const uint32_t mc = 19250u, kvh = 4u, hd = 64u;
        QwenKvLayoutDesc fd{}, bd{};
        check(qwen_kv_layout_desc(QwenKvDType::f32, QwenKvLayout::position_major, mc, kvh, hd, fd).ok(),
              "kv contract: f32 position-major desc");
        check(qwen_kv_layout_desc(QwenKvDType::bf16, QwenKvLayout::blocked64, mc, kvh, hd, bd).ok(),
              "kv contract: bf16 blocked64 desc");
        const uint64_t pb = (mc + 63u) / 64u;
        check(bd.position_blocks == pb && pb == 301u &&
                  bd.head_stride_codes == pb * 64u * hd &&
                  bd.region_codes == kvh * pb * 64u * hd &&
                  bd.region_words * 2u == bd.region_codes,
              "kv contract: blocked region is heads x ceil(ctx/64) x 64 x head_dim codes");
        check(fd.region_words == static_cast<uint64_t>(mc) * kvh * hd &&
                  bd.payload_codes() == fd.region_words &&
                  bd.padding_codes() == static_cast<uint64_t>(kvh) * (pb * 64u - mc) * hd &&
                  bd.padding_codes() == static_cast<uint64_t>(kvh) * 14u * hd,
              "kv contract: payload excludes the 64-position padding and padding is accounted");
        check(bd.bytes_per_token() == 2u * kvh * hd && fd.bytes_per_token() == 4u * kvh * hd &&
                  bd.payload_bytes() * 2u == fd.payload_bytes(),
              "kv contract: bf16 stores exactly half the bytes per token");
        // An odd head_dim cannot pack two codes per word without straddling a row.
        QwenKvLayoutDesc bad{};
        check(qwen_kv_layout_desc(QwenKvDType::bf16, QwenKvLayout::blocked64, mc, 1u, 3u, bad).code ==
                  StatusCode::invalid_argument,
              "kv contract: bf16 refuses an odd head_dim (fails closed)");

        // Address injectivity + bounds over EVERY (head, position, channel) of a
        // ragged context: a layout that truncates, wraps or aliases fails here.
        {
            luisa::vector<uint8_t> seen(bd.region_words * 2u, 0u);// one byte per CODE
            bool injective = true, in_range = true;
            for (uint32_t h = 0; h < kvh && injective; h++) {
                for (uint32_t p = 0; p < mc && injective; p++) {
                    for (uint32_t c = 0; c < hd; c++) {
                        const auto code = bd.code_index(h, p, c);
                        const auto w = bd.word_index(code);
                        const auto shift = bd.word_shift(code);
                        in_range = in_range && w < bd.region_words && (shift == 0u || shift == 16u);
                        const auto slot = w * 2u + (shift != 0u ? 1u : 0u);
                        if (slot >= bd.region_words * 2u || seen[slot] != 0u) {
                            injective = false;
                            break;
                        }
                        seen[slot] = 1u;
                    }
                }
            }
            const uint64_t covered = static_cast<uint64_t>(kvh) * mc * hd;
            check(injective && in_range && covered == bd.payload_codes(),
                  "kv contract: every (head, position, channel) maps to a distinct in-range code");
            // Block boundaries: position 63 -> block 0 row 63, position 64 ->
            // block 1 row 0, and the two are one whole block apart.
            const auto at63 = bd.code_index(1u, 63u, 0u);
            const auto at64 = bd.code_index(1u, 64u, 0u);
            check(at64 - at63 == hd && at64 == bd.head_base_codes(1u) + bd.block_codes(),
                  "kv contract: 64-position block boundaries are exact");
        }

        // What one page covers, in TOKEN terms, per dtype/layout: the number the
        // caps log and the docs quote. 2 MiB / (1024 values/token x 4 B) = 512 at
        // f32, 1024 at bf16 in the LEGACY geometry; the blocked layout covers
        // page_bytes / (head_dim x 2 B) POSITIONS OF ONE HEAD per page, which is
        // why the first-token residency is one page per head.
        const uint64_t page_words = 2097152ull / 4ull;
        QwenKvLayoutDesc model_f{}, model_b{};
        check(qwen_kv_layout_desc(QwenKvDType::f32, QwenKvLayout::position_major, 4096u, 4u, 256u, model_f).ok() &&
                  qwen_kv_layout_desc(QwenKvDType::bf16, QwenKvLayout::blocked64, 4096u, 4u, 256u, model_b).ok(),
              "kv contract: model geometry descs");
        check(page_words / model_f.legacy_codes_per_token() == 512u &&
                  (page_words * 2u) / model_b.legacy_codes_per_token() == 1024u,
              "kv contract: a 2 MiB page covers 512 fp32 / 1024 bf16 positions per token of payload");
        check(model_b.positions_per_page_per_head(page_words) == 4096u &&
                  model_b.head_stride_codes == 4096ull * 256ull,
              "kv contract: at the model geometry one bf16 page covers all 4096 positions of one head");

        // ------------------------------------------------------------------
        // fp8_e4m3 KV contract (opt_plan/05 Q3, FP8 step): the weight codec, the
        // per-(kv_head, 64-position block) scale axis, the ONLINE APPEND rule and
        // the fail-closed geometry. Everything below is host-only and is the same
        // arithmetic the device kernels implement (checked bit for bit by the DSL
        // suite), so a disagreement is caught before any kernel runs.
        // ------------------------------------------------------------------
        {
            // Codec: E4M3FN, 448 saturation, RNE, no infinities, NaN = 0x7f.
            check(qwen_kv_fp8_code_to_f32(0x00u) == 0.0f &&
                      qwen_kv_fp8_code_to_f32(0x80u) == 0.0f &&
                      std::signbit(qwen_kv_fp8_code_to_f32(0x80u)),
                  "fp8 contract: the two zero codes decode to +0 and -0");
            check(qwen_kv_fp8_code_to_f32(0x01u) == std::ldexp(1.0f, -9) &&
                      qwen_kv_fp8_code_to_f32(0x07u) == 7.0f * std::ldexp(1.0f, -9) &&
                      qwen_kv_fp8_code_to_f32(0x08u) == std::ldexp(1.0f, -6),
                  "fp8 contract: subnormal codes are k*2^-9 and 0x08 is the smallest normal");
            check(qwen_kv_fp8_code_to_f32(0x7Eu) == 448.0f &&
                      qwen_kv_fp8_code_to_f32(0xFEu) == -448.0f &&
                      std::isnan(qwen_kv_fp8_code_to_f32(0x7Fu)) &&
                      std::isnan(qwen_kv_fp8_code_to_f32(0xFFu)),
                  "fp8 contract: 0x7e is 448 (the largest finite), 0x7f/0xff are NaN");
            check(qwen_kv_fp8_code_from_f32(448.0f) == 0x7Eu &&
                      qwen_kv_fp8_code_from_f32(1.0e30f) == 0x7Eu &&
                      qwen_kv_fp8_code_from_f32(-1.0e30f) == 0xFEu &&
                      qwen_kv_fp8_code_from_f32(std::numeric_limits<float>::infinity()) == 0x7Eu,
                  "fp8 contract: saturation clamps every overflow to +/-448 (no infinity code)");
            check(qwen_kv_fp8_code_from_f32(std::numeric_limits<float>::quiet_NaN()) == 0x7Fu,
                  "fp8 contract: NaN maps to the single NaN code 0x7f");
            // RNE ties, both parities: 0.5 unit rounds to 0 (even), 1.5 units to 2,
            // 2.5 units to 2 (even), 3.5 units to 4.
            const float unit = std::ldexp(1.0f, -9);
            check(qwen_kv_fp8_code_from_f32(0.5f * unit) == 0x00u &&
                      qwen_kv_fp8_code_from_f32(1.5f * unit) == 0x02u &&
                      qwen_kv_fp8_code_from_f32(2.5f * unit) == 0x02u &&
                      qwen_kv_fp8_code_from_f32(3.5f * unit) == 0x04u,
                  "fp8 contract: ties round to even in the subnormal range");
            check(qwen_kv_fp8_code_from_f32(1.0f + 1.0f / 16.0f) == 0x38u &&
                      qwen_kv_fp8_code_from_f32(1.0f + 3.0f / 16.0f) == 0x3Au,
                  "fp8 contract: ties round to even in the normal range");
            // A 256-code round trip: decode(encode(decode(code))) == decode(code) for
            // every finite code, i.e. the encoder never moves a representable value.
            bool round_trip = true;
            for (uint32_t code = 0u; code < 256u; ++code) {
                const auto v = qwen_kv_fp8_code_to_f32(static_cast<uint8_t>(code));
                if (std::isnan(v)) { continue; }
                const auto back = qwen_kv_fp8_code_from_f32(v);
                round_trip = round_trip && qwen_kv_fp8_code_to_f32(back) == v;
            }
            check(round_trip, "fp8 contract: every finite code survives decode->encode unchanged");

            // -- geometry: fp8 requires the blocked layout and head_dim % 4 == 0 --
            const uint32_t mc8 = 200u, kvh8 = 2u, hd8 = 8u;
            QwenKvLayoutDesc d8{};
            check(qwen_kv_layout_desc(QwenKvDType::fp8_e4m3, QwenKvLayout::blocked64, mc8, kvh8, hd8, d8).ok(),
                  "fp8 contract: blocked desc");
            check(d8.elements_per_word() == 4u && d8.scaled() && d8.position_blocks == 4u &&
                      d8.scale_count == kvh8 * 4u && d8.scale_bytes() == kvh8 * 4u * sizeof(float),
                  "fp8 contract: four codes per word and one scale per (head, block)");
            check(d8.payload_bytes() * 4u ==
                      static_cast<uint64_t>(mc8) * kvh8 * hd8 * 4u &&
                      d8.bytes_per_token() == kvh8 * hd8,
                  "fp8 contract: payload is exactly a quarter of the FP32 bytes");
            QwenKvLayoutDesc bad8{};
            check(qwen_kv_layout_desc(QwenKvDType::fp8_e4m3, QwenKvLayout::position_major,
                                      mc8, kvh8, hd8, bad8).code == StatusCode::invalid_argument,
                  "fp8 contract: position_major is refused (a block scale needs position blocks)");
            check(qwen_kv_layout_desc(QwenKvDType::fp8_e4m3, QwenKvLayout::blocked64,
                                      mc8, kvh8, 6u, bad8).code == StatusCode::invalid_argument,
                  "fp8 contract: head_dim % 4 != 0 is refused (a quad must not straddle a row)");
            check(qwen_kv_layout_desc(QwenKvDType::fp8_e4m3, QwenKvLayout::blocked64,
                                      0xFFFFFFFFu, 0xFFFFFFFFu, 4u, bad8).code ==
                      StatusCode::out_of_memory,
                  "fp8 contract: an overflowing scale array is refused (not truncated)");
            check(d8.scale_index(0u, 0u) == 0u && d8.scale_index(0u, 63u) == 0u &&
                      d8.scale_index(0u, 64u) == 1u && d8.scale_index(1u, 199u) == 4u + 3u,
                  "fp8 contract: scale index = head * position_blocks + position / 64");
            check(qwen_kv_block_scale(448.0f) == 1.0f && qwen_kv_block_scale(0.0f) == 0.0f &&
                      qwen_kv_block_limit(1.0f) == 448.0f,
                  "fp8 contract: block scale = max_abs / 448 and its limit is 448*scale");
            check(qwen_kv_counts_for_max(1.0f) && !qwen_kv_counts_for_max(std::nanf("")) &&
                      !qwen_kv_counts_for_max(std::numeric_limits<float>::infinity()),
                  "fp8 contract: non-finite values never participate in a block maximum");

            // ---- online append: growth, non-growth, boundaries, ragged tail ----
            const auto decode_region = [&](const QwenKvLayoutDesc &d, luisa::span<const uint32_t> w,
                                           luisa::span<const float> sc, uint32_t tokens) {
                luisa::vector<float> dense(static_cast<size_t>(tokens) * d.legacy_codes_per_token(), 0.0f);
                const auto st = qwen_kv_region_to_dense(d, w, sc, tokens, dense);
                return std::make_pair(st, dense);
            };
            const auto values_at = [](const luisa::vector<float> &dense, const QwenKvLayoutDesc &d,
                                      uint32_t head, uint32_t pos, uint32_t channel) {
                return dense[static_cast<size_t>(pos) * d.legacy_codes_per_token() +
                             static_cast<size_t>(head) * d.head_dim + channel];
            };
            luisa::vector<uint32_t> words(static_cast<size_t>(d8.region_words), 0u);
            luisa::vector<float> scales(static_cast<size_t>(d8.scale_count), 0.0f);
            check([&] {
                for (const auto s : scales) {
                    if (s != 0.0f) { return false; }
                }
                return true;
            }(),
                  "fp8 append: a fresh region starts with every scale 0 (never written)");

            // Chunk 1: positions 0..9, all small (max |v| = 1) -> scale = 1/448.
            luisa::vector<float> src1(static_cast<size_t>(10u) * d8.legacy_codes_per_token());
            for (uint32_t t = 0u; t < 10u; t++) {
                for (uint32_t i = 0u; i < d8.legacy_codes_per_token(); i++) {
                    src1[static_cast<size_t>(t) * d8.legacy_codes_per_token() + i] =
                        ((t + i) % 2u == 0u ? 1.0f : -1.0f) * (0.5f + 0.05f * static_cast<float>(i % 7u));
                }
            }
            check(qwen_kv_append_reference(d8, src1, 0u, 10u, scales, words).ok(),
                  "fp8 append: chunk 1 (positions 0..9)");
            // The fixture's largest magnitude is 0.5 + 0.05*6 = 0.8.
            check(scales[d8.scale_index(0u, 0u)] == qwen_kv_block_scale(0.8f) &&
                      scales[d8.scale_index(1u, 0u)] == qwen_kv_block_scale(0.8f) &&
                      scales[d8.scale_index(0u, 64u)] == 0.0f,
                  "fp8 append: only the touched block gets a scale, and it is max_abs/448");
            const auto snapshot1 = words;
            const auto scale1 = scales;
            {
                const auto [st, dense] = decode_region(d8, words, scales, 10u);
                check(st.ok(), "fp8 append: dequantize chunk 1");
                float worst = 0.0f;
                for (uint32_t t = 0u; t < 10u; t++) {
                    for (uint32_t i = 0u; i < d8.legacy_codes_per_token(); i++) {
                        const auto want = src1[static_cast<size_t>(t) * d8.legacy_codes_per_token() + i];
                        worst = std::max(worst, std::fabs(want - dense[static_cast<size_t>(t) * d8.legacy_codes_per_token() + i]));
                    }
                }
                // E4M3 has 3 mantissa bits: RNE error <= half a ulp = 2^-4 of the
                // value, i.e. <= 6.25 % of the block maximum.
                check(worst <= 0.0625f * 0.8f + 1e-7f,
                      "fp8 append: chunk 1 dequantizes within half an E4M3 ulp of the block max");
            }
            // Chunk 2: positions 10..19 with a 10x larger magnitude -> the block
            // maximum GROWS, so every earlier code must be rescaled against the new
            // scale (never left paired with it).
            luisa::vector<float> src2(static_cast<size_t>(10u) * d8.legacy_codes_per_token());
            for (uint32_t t = 0u; t < 10u; t++) {
                for (uint32_t i = 0u; i < d8.legacy_codes_per_token(); i++) {
                    src2[static_cast<size_t>(t) * d8.legacy_codes_per_token() + i] =
                        (i % 2u == 0u ? 10.0f : -10.0f);
                }
            }
            check(qwen_kv_append_reference(d8, src2, 10u, 10u, scales, words).ok(),
                  "fp8 append: chunk 2 (positions 10..19, 10x the earlier maximum)");
            check(scales[d8.scale_index(0u, 0u)] == qwen_kv_block_scale(10.0f) &&
                      scales[d8.scale_index(0u, 0u)] > scale1[d8.scale_index(0u, 0u)],
                  "fp8 append: the block scale grew to max_abs/448");
            check(words != snapshot1, "fp8 append: growing the maximum REWRITES the earlier codes");
            {
                const auto [st, dense] = decode_region(d8, words, scales, 20u);
                check(st.ok(), "fp8 append: dequantize after growth");
                // The EARLIER positions must still hold their own values (within the
                // new, coarser block step), i.e. they were rescaled, not corrupted.
                float worst_old = 0.0f;
                for (uint32_t t = 0u; t < 10u; t++) {
                    for (uint32_t i = 0u; i < d8.legacy_codes_per_token(); i++) {
                        const auto want = src1[static_cast<size_t>(t) * d8.legacy_codes_per_token() + i];
                        worst_old = std::max(worst_old,
                                             std::fabs(want - dense[static_cast<size_t>(t) * d8.legacy_codes_per_token() + i]));
                    }
                }
                check(worst_old <= 0.0625f * 10.0f + 1e-6f,
                      "fp8 append: the earlier values survive the rescale within the new step");
                // And the exact invariant: value == code * scale for the code stored.
                bool exact = true;
                for (uint32_t head = 0u; head < kvh8 && exact; head++) {
                    for (uint32_t pos = 0u; pos < 20u && exact; pos++) {
                        const auto s = scales[d8.scale_index(head, pos)];
                        for (uint32_t c = 0u; c < hd8 && exact; c++) {
                            const auto code = d8.code_index(head, pos, c);
                            const auto byte = static_cast<uint8_t>(
                                (words[d8.word_index(code)] >> d8.word_shift(code)) & 0xFFu);
                            exact = qwen_kv_fp8_code_to_f32(byte) * s ==
                                    values_at(dense, d8, head, pos, c);
                        }
                    }
                }
                check(exact, "fp8 append: every read value is exactly code * its own block scale");
            }
            // Chunk 3: positions 20..29 stay BELOW the block maximum -> the scale
            // must not change and the earlier bytes must stay byte-identical.
            luisa::vector<float> src3(static_cast<size_t>(10u) * d8.legacy_codes_per_token());
            for (auto &v : src3) { v = 2.0f; }
            const auto snapshot2 = words;
            const auto scale2 = scales;
            check(qwen_kv_append_reference(d8, src3, 20u, 10u, scales, words).ok(),
                  "fp8 append: chunk 3 (positions 20..29, below the block maximum)");
            check(scales == scale2, "fp8 append: a non-growing append keeps every scale");
            {
                bool prefix_intact = true;
                for (uint32_t head = 0u; head < kvh8; head++) {
                    for (uint32_t pos = 0u; pos < 20u; pos++) {
                        for (uint32_t c = 0u; c < hd8; c += 4u) {
                            const auto code = d8.code_index(head, pos, c);
                            prefix_intact = prefix_intact &&
                                            words[d8.word_index(code)] == snapshot2[d8.word_index(code)];
                        }
                    }
                }
                check(prefix_intact, "fp8 append: a non-growing append leaves earlier words untouched");
                const auto [st, dense] = decode_region(d8, words, scales, 30u);
                check(st.ok(), "fp8 append: dequantize after the non-growing append");
                check(std::fabs(values_at(dense, d8, 0u, 25u, 0u) - 2.0f) <=
                          0.0625f * 2.0f + 1e-6f,
                      "fp8 append: the non-growing values are stored at the block scale");
            }
            // Chunk 4: a chunk that CROSSES a 64-position block boundary touches two
            // blocks, each with its own scale.
            luisa::vector<float> src4(static_cast<size_t>(7u) * d8.legacy_codes_per_token());
            for (uint32_t t = 0u; t < 7u; t++) {
                for (uint32_t i = 0u; i < d8.legacy_codes_per_token(); i++) {
                    src4[static_cast<size_t>(t) * d8.legacy_codes_per_token() + i] =
                        (t == 6u ? 100.0f : 0.25f);
                }
            }
            check(qwen_kv_append_reference(d8, src4, 60u, 7u, scales, words).ok(),
                  "fp8 append: chunk 4 crosses the 64-position block boundary");
            check(scales[d8.scale_index(0u, 60u)] == qwen_kv_block_scale(10.0f) &&
                      scales[d8.scale_index(0u, 64u)] == qwen_kv_block_scale(100.0f) &&
                      scales[d8.scale_index(1u, 64u)] == qwen_kv_block_scale(100.0f),
                  "fp8 append: each crossed block derives its own scale from its own values");
            // Ragged tail: positions 192..199 live in the last (padded) block.
            {
                luisa::vector<float> src5(static_cast<size_t>(8u) * d8.legacy_codes_per_token());
                for (uint32_t t = 0u; t < 8u; t++) {
                    for (uint32_t i = 0u; i < d8.legacy_codes_per_token(); i++) {
                        src5[static_cast<size_t>(t) * d8.legacy_codes_per_token() + i] =
                            static_cast<float>(t) - 3.0f;
                    }
                }
                check(qwen_kv_append_reference(d8, src5, 192u, 8u, scales, words).ok(),
                      "fp8 append: ragged tail (positions 192..199 of a 200-position block)");
                const auto [st, dense] = decode_region(d8, words, scales, 200u);
                check(st.ok() && std::fabs(values_at(dense, d8, 1u, 199u, 7u) - 4.0f) <=
                                     3.0f / 448.0f * 0.0625f + 1e-6f,
                      "fp8 append: the last valid position of the ragged tail is exact");
            }
            // reset/reuse: zeroing words+scales then replaying the same chunks must
            // reproduce the same bytes (the engine's reset() does exactly this).
            {
                luisa::vector<uint32_t> w2(static_cast<size_t>(d8.region_words), 0u);
                luisa::vector<float> s2(static_cast<size_t>(d8.scale_count), 0.0f);
                luisa::vector<float> src5(static_cast<size_t>(8u) * d8.legacy_codes_per_token());
                for (uint32_t t = 0u; t < 8u; t++) {
                    for (uint32_t i = 0u; i < d8.legacy_codes_per_token(); i++) {
                        src5[static_cast<size_t>(t) * d8.legacy_codes_per_token() + i] =
                            static_cast<float>(t) - 3.0f;
                    }
                }
                check(qwen_kv_append_reference(d8, src1, 0u, 10u, s2, w2).ok() &&
                          qwen_kv_append_reference(d8, src2, 10u, 10u, s2, w2).ok() &&
                          qwen_kv_append_reference(d8, src3, 20u, 10u, s2, w2).ok() &&
                          qwen_kv_append_reference(d8, src4, 60u, 7u, s2, w2).ok() &&
                          qwen_kv_append_reference(d8, src5, 192u, 8u, s2, w2).ok() &&
                          w2 == words && s2 == scales,
                      "fp8 append: a reset + replay of every chunk reproduces the exact words and scales");
            }
            // Dense -> region is the SAME rule (it forwards to the append oracle),
            // so a cache built from a dense fixture is append-consistent.
            {
                luisa::vector<float> dense_all(static_cast<size_t>(200u) * d8.legacy_codes_per_token());
                for (size_t i = 0; i < dense_all.size(); i++) {
                    dense_all[i] = 0.01f * static_cast<float>(static_cast<int>(i % 97u) - 48);
                }
                luisa::vector<uint32_t> w3(static_cast<size_t>(d8.region_words), 0u);
                luisa::vector<float> s3(static_cast<size_t>(d8.scale_count), 0.0f);
                luisa::vector<float> back(dense_all.size(), 0.0f);
                check(qwen_kv_region_from_dense(d8, s3, dense_all, 200u, w3).ok() &&
                          qwen_kv_region_to_dense(d8, w3, s3, 200u, back).ok(),
                      "fp8 contract: dense -> region -> dense round trip");
                float worst = 0.0f;
                for (size_t i = 0; i < dense_all.size(); i++) {
                    worst = std::max(worst, std::fabs(dense_all[i] - back[i]));
                }
                check(worst <= 0.0625f * 0.01f * 48.0f + 1e-7f,
                      "fp8 contract: the round trip is inside half an E4M3 ulp of the block maximum");
                // An unscaled read of an fp8 region must be REFUSED, never silently
                // interpreted as float bytes.
                check(!qwen_kv_region_to_dense(d8, w3, 200u, back).ok(),
                      "fp8 contract: dequantizing a scaled region without its scales is refused");
            }
            // Non-finite source values: excluded from the block maximum (so the block
            // keeps its useful scale), stored as their E4M3 code at their own position.
            {
                luisa::vector<uint32_t> w4(static_cast<size_t>(d8.region_words), 0u);
                luisa::vector<float> s4(static_cast<size_t>(d8.scale_count), 0.0f);
                luisa::vector<float> src6(static_cast<size_t>(2u) * d8.legacy_codes_per_token(), 1.0f);
                const auto row = d8.legacy_codes_per_token();
                src6[0] = std::numeric_limits<float>::quiet_NaN();
                src6[1] = std::numeric_limits<float>::infinity();
                check(qwen_kv_append_reference(d8, src6, 0u, 2u, s4, w4).ok(),
                      "fp8 append: NaN/inf rows are accepted");
                  // The append rule maps a non-finite value to ONE FULL-SCALE UNIT
                  // (qwen_kv_counts_for_max, kv_cache_layout.h), the same rule the
                  // device kernel applies (dsl_ops.cpp), so invalid data can neither
                  // zero the block scale (0/0) nor inflate it to infinity. A block
                  // that mixes NaN/inf with finite rows therefore ends at the
                  // full-scale unit (scale 1), and every finite row is still exact
                  // there. The earlier revision of this arm asserted the superseded
                  // "excluded from the maximum" rule (scale 1/448).
                  check(std::isfinite(s4[d8.scale_index(0u, 0u)]) &&
                            s4[d8.scale_index(0u, 0u)] == qwen_kv_block_scale(kFp8E4M3Max),
                        "fp8 append: a non-finite row contributes one full-scale unit, "
                        "never zero and never infinity");
                const auto code = d8.code_index(0u, 0u, 0u);
                const auto byte = static_cast<uint8_t>((w4[d8.word_index(code)] >> d8.word_shift(code)) & 0xFFu);
                check(byte == 0x7Fu && std::isnan(qwen_kv_fp8_code_to_f32(byte)),
                      "fp8 append: a NaN is stored as the NaN code at its own position");
                  const auto code_inf = d8.code_index(0u, 0u, 1u);
                  const auto byte_inf = static_cast<uint8_t>((w4[d8.word_index(code_inf)] >> d8.word_shift(code_inf)) & 0xFFu);
                  check(byte_inf == 0x7Eu, "fp8 append: an infinity saturates to 448");
                  // The finite row of the same block is still exact at that scale
                  // (1.0 is representable at scale 1), so the full-scale unit does
                  // not corrupt the valid data that shares the block.
                  const auto [st6, dense6] = decode_region(d8, w4, s4, 2u);
                  check(st6.ok() && values_at(dense6, d8, 0u, 1u, 0u) == 1.0f,
                        "fp8 append: the finite row of a non-finite block stays exact");
            }
        }

    }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    if (!device) {
        std::fprintf(stderr, "failed to create device for backend '%s'\n", argv[1]);
        return 2;
    }
    auto stream = device.create_stream();
    auto shader = device.compile(make_write_kv_kernel());
    auto window_shader = device.compile(make_write_window_kernel());

    // ------------------------------------------------------------------
    // Stage 1: probe the device tile granularity and cover destruction
    // after a dispatched write (no synchronize before the destructor).
    // ------------------------------------------------------------------
    constexpr uint64_t kMinPageBytes = 2ull * 1024ull * 1024ull;
    uint32_t tile_bytes = 0u;
    {
        QwenKvCacheAllocator probe;
        auto st = probe.create(device, stream, 8u, 4u, 1u);
        if (!st.ok()) {
            if (st.code == StatusCode::unsupported) {
                std::printf("SKIP: backend '%s' has no sparse-buffer support (%s)\n",
                            argv[1], st.message.c_str());
                return 0;
            }
            check(false, "probe allocator create", st.message);
            return summarize("qwen kv cache", argv[1]);
        }
        check(true, "probe allocator create");
        tile_bytes = probe.tile_size_bytes();
        check(tile_bytes >= sizeof(float) && tile_bytes % sizeof(float) == 0u,
              "probe tile is a whole number of floats");
        check(probe.committed_bytes() == 0u && probe.mapped_heap_count() == 0u &&
                  probe.committed_tokens() == 0u,
              "probe starts with zero residency");
        check(probe.ensure_capacity(stream, 2u).ok(), "probe ensure_capacity");
        // The page model allocates ONE heap per page and maps tiles_per_page
        // tiles out of it, so residency is counted in pages, not in tiles. Two
        // tokens of four floats fit in the first page of every region.
        const uint32_t probe_pages = probe.committed_pages_per_region();
        check(probe.committed_tokens() == 2u && probe_pages == 1u &&
                  probe.mapped_heap_count() == 2u * probe_pages &&
                  probe.committed_bytes() ==
                      static_cast<uint64_t>(2u * probe_pages) * probe.page_size_bytes(),
              "probe commits exactly the required page heaps");
        const uint32_t probe_write = std::min(8u, 2u * 4u);
        dispatch_write(shader, stream, probe.k_view(0), probe.v_view(0), probe_write, 0.25f, 0.125f);
        // probe destroyed here, in flight, before any synchronize.
    }
    stream.synchronize();
    check(true, "destructor after in-flight dispatch (probe)");

    // ------------------------------------------------------------------
    // Stage 2: main allocator. Every number is derived from the probed
    // tile size, so the test crosses a real PAGE boundary on any backend:
    // page_bytes follows the documented formula, and one page holds
    // page_elems/vpt tokens of `vpt` values per token.
    // ------------------------------------------------------------------
    const uint32_t vpt = 12u;  // ragged: 48 B/token vs power-of-two tiles
    const uint32_t layers = 3u;// three K/V pairs
    const uint32_t tile_elems = tile_bytes / sizeof(float);
    const uint64_t page_bytes = ceil_div(std::max(kMinPageBytes,
                                                  static_cast<uint64_t>(tile_bytes)),
                                         tile_bytes) *
                                tile_bytes;
    const uint32_t page_elems = static_cast<uint32_t>(page_bytes / sizeof(float));
    // Tokens covered by one page inside a region (floor: the page boundary falls
    // inside a token for ragged strides, which is exactly the interesting case).
    const uint32_t tokens_per_page = std::max(1u, page_elems / vpt);
    const uint32_t cross_tokens = tokens_per_page + 3u;      // crosses ONE page boundary
    const uint32_t max_context = cross_tokens + tokens_per_page / 2u;// + a partial page
    const uint32_t region_count = 2u * layers;
    // Expected residency of the configuration under test, expressed in pages.
    const auto expected_pages = [&](uint32_t tokens) {
        const uint64_t bytes = static_cast<uint64_t>(tokens) * vpt * sizeof(float);
        return static_cast<uint32_t>(std::min<uint64_t>(ceil_div(bytes, page_bytes),
                                                       ceil_div(static_cast<uint64_t>(max_context) * vpt * sizeof(float),
                                                                page_bytes)));
    };
    const auto expected_heaps = [&](uint32_t tokens) {
        return region_count * expected_pages(tokens);
    };

    auto alloc = luisa::make_unique<QwenKvCacheAllocator>();
    {
        auto st = alloc->create(device, stream, max_context, vpt, layers);
        check(st.ok(), "allocator create", st.message);
        if (!st.ok()) { return summarize("qwen kv cache", argv[1]); }
    }
    check(alloc->valid(), "allocator valid after create");
    check(alloc->max_context() == max_context && alloc->values_per_token() == vpt &&
              alloc->layer_count() == layers,
          "configured geometry accessors");
    check(alloc->committed_bytes() == 0u && alloc->mapped_heap_count() == 0u &&
              alloc->committed_tokens() == 0u,
          "zero residency after create");
    check(alloc->reset(stream).ok() && alloc->valid() &&
              alloc->committed_bytes() == 0u && alloc->mapped_heap_count() == 0u,
          "reset with no committed tiles preserves the empty buffer");
    check(alloc->logical_bytes() ==
              static_cast<uint64_t>(region_count) * max_context * vpt * sizeof(float),
          "logical_bytes is the exact configured payload");
    check(alloc->reserved_bytes() >= alloc->logical_bytes() &&
              alloc->reserved_bytes() % tile_bytes == 0u,
          "reserved_bytes is the tile-padded virtual size");
    check(alloc->cache_stride_elements() >= static_cast<uint64_t>(max_context) * vpt &&
              alloc->cache_stride_elements() % page_elems == 0u,
          "region stride is page-aligned");

    // ---- page-model invariants (spec checks 1 and 2) --------------------
    {
        const uint32_t page_size = alloc->page_size_bytes();
        const uint32_t tiles_per_page = alloc->tiles_per_page();
        const uint32_t pages_per_region = alloc->pages_per_region();
        const uint64_t logical_elems = static_cast<uint64_t>(max_context) * vpt;
        check(page_size >= kMinPageBytes && page_size % tile_bytes == 0u,
              "page_size_bytes() is >= 2 MiB and an exact multiple of the tile");
        check(tiles_per_page == page_size / tile_bytes && tiles_per_page >= 1u,
              "tiles_per_page() == page_size_bytes()/tile_size_bytes()");
        check(pages_per_region == ceil_div(logical_elems * sizeof(float), page_size),
              "pages_per_region() == ceil(logical bytes / page bytes)");
        check(alloc->cache_stride_elements() ==
                  static_cast<uint64_t>(pages_per_region) * (page_size / sizeof(float)),
              "cache_stride_elements() == pages_per_region * page elements");
        check(alloc->heap_slot_count() == region_count * pages_per_region,
              "heap_slot_count() == region_count * pages_per_region");
        check(alloc->reserved_bytes() ==
                  static_cast<uint64_t>(alloc->heap_slot_count()) * page_size,
              "reserved_bytes() == heap slots * page size");
        check(expected_pages(max_context) == pages_per_region,
              "spec page formula reproduces the allocator geometry");
    }

    // View geometry: exact dense sizes, distinct page-aligned region bases.
    {
        bool sizes_ok = true, bases_ok = true, aligned_ok = true;
        luisa::vector<uint64_t> bases;
        for (uint32_t l = 0u; l < layers; ++l) {
            const auto k = alloc->k_view(l);
            const auto v = alloc->v_view(l);
            sizes_ok = sizes_ok && k.size() == max_context * vpt && v.size() == max_context * vpt;
            bases.emplace_back(k.offset());
            bases.emplace_back(v.offset());
            aligned_ok = aligned_ok && k.offset() % page_elems == 0u &&
                         v.offset() % page_elems == 0u;
        }
        std::sort(bases.begin(), bases.end());
        bases_ok = std::adjacent_find(bases.begin(), bases.end()) == bases.end();
        check(sizes_ok, "k/v views have the exact dense logical size");
        check(bases_ok, "region bases are pairwise distinct");
        check(aligned_ok, "region bases are page-aligned");
    }

    // On-demand commit, idempotence, and argument validation.
    check(alloc->ensure_capacity(stream, 0u).ok(), "ensure_capacity(0) is a no-op");
    check(alloc->ensure_capacity(stream, 3u).ok(), "first ensure_capacity");
    const uint64_t small_commit = alloc->committed_bytes();
    check(small_commit > 0u && alloc->mapped_heap_count() == expected_heaps(3u) &&
              alloc->committed_pages_per_region() == expected_pages(3u) &&
              alloc->committed_tokens() == 3u,
          "ensure_capacity commits exactly the required page heaps");
    check(alloc->ensure_capacity(stream, 3u).ok() &&
              alloc->ensure_capacity(stream, 0u).ok() &&
              alloc->committed_bytes() == small_commit &&
              alloc->mapped_heap_count() == expected_heaps(3u),
          "ensure_capacity is idempotent below the committed level");
    {
        auto st = alloc->ensure_capacity(stream, max_context + 1u);
        check(!st.ok() && st.code == StatusCode::invalid_argument,
              "ensure_capacity rejects tokens > max_context");
        check(alloc->committed_bytes() == small_commit,
              "rejected ensure_capacity changes nothing");
    }
    {
        auto other_stream = device.create_stream();
        auto grow = alloc->ensure_capacity(other_stream, 4u);
        check(!grow.ok() && grow.code == StatusCode::invalid_argument &&
                  alloc->committed_bytes() == small_commit,
              "ensure_capacity rejects a non-creation stream");
        auto reset = alloc->reset(other_stream);
        check(!reset.ok() && reset.code == StatusCode::invalid_argument &&
                  alloc->valid() && alloc->committed_bytes() == small_commit,
              "reset rejects a non-creation stream without unmapping");
    }

    // Early data on layers 0 and 2, then growth that CROSSES a page boundary
    // (page 0 -> page 1) and therefore adds one heap per region.
    const uint32_t t_small = 3u;
    dispatch_write(shader, stream, alloc->k_view(0), alloc->v_view(0),
                   t_small * vpt, 0.25f, 0.125f);
    dispatch_write(shader, stream, alloc->k_view(2), alloc->v_view(2),
                   t_small * vpt, -0.5f, 2.0f);
    stream.synchronize();
    {
        const auto k0 = read_back(stream, alloc->k_view(0).subview(0u, t_small * vpt));
        const auto v0 = read_back(stream, alloc->v_view(0).subview(0u, t_small * vpt));
        const auto k2 = read_back(stream, alloc->k_view(2).subview(0u, t_small * vpt));
        const auto v2 = read_back(stream, alloc->v_view(2).subview(0u, t_small * vpt));
        check_exact(k0, pattern(t_small * vpt, 0.25f), "layer 0 K pattern (pre-growth)");
        check_exact(v0, pattern(t_small * vpt, 0.125f), "layer 0 V pattern (pre-growth)");
        check_exact(k2, pattern(t_small * vpt, -0.5f), "layer 2 K pattern (pre-growth)");
        check_exact(v2, pattern(t_small * vpt, 2.0f), "layer 2 V pattern (pre-growth)");
    }

    const auto k0_offset_before = alloc->k_view(0).offset();
    check(alloc->ensure_capacity(stream, cross_tokens).ok(),
          "growth across a page boundary");
    check(alloc->committed_tokens() == cross_tokens &&
              alloc->committed_pages_per_region() == expected_pages(cross_tokens) &&
              alloc->mapped_heap_count() == expected_heaps(cross_tokens) &&
              alloc->committed_bytes() >
                  static_cast<uint64_t>(expected_heaps(3u)) * alloc->page_size_bytes() &&
              alloc->committed_bytes() ==
                  static_cast<uint64_t>(alloc->mapped_heap_count()) * alloc->page_size_bytes(),
          "growth increases residency by one page per region");
    check(alloc->k_view(0).offset() == k0_offset_before &&
              alloc->k_view(0).size() == max_context * vpt &&
              alloc->v_view(2).offset() > alloc->k_view(2).offset(),
          "views are unchanged by growth");
    {
        // Early data must survive the new page mappings untouched: the first
        // page keeps its heap and its contents across the growth step.
        const auto k0 = read_back(stream, alloc->k_view(0).subview(0u, t_small * vpt));
        const auto k2 = read_back(stream, alloc->k_view(2).subview(0u, t_small * vpt));
        check_exact(k0, pattern(t_small * vpt, 0.25f), "layer 0 K preserved after growth");
        check_exact(k2, pattern(t_small * vpt, -0.5f), "layer 2 K preserved after growth");
    }

    // Full-span rewrite on layers 0 and 1 covering positions that cross the
    // page boundary inside a region; exact readback of the whole span.
    dispatch_write(shader, stream, alloc->k_view(0), alloc->v_view(0),
                   cross_tokens * vpt, 0.25f, 0.125f);
    dispatch_write(shader, stream, alloc->k_view(1), alloc->v_view(1),
                   cross_tokens * vpt, 4.0f, -0.0625f);
    stream.synchronize();
    {
        const auto k0 = read_back(stream, alloc->k_view(0).subview(0u, cross_tokens * vpt));
        const auto v0 = read_back(stream, alloc->v_view(0).subview(0u, cross_tokens * vpt));
        const auto k1 = read_back(stream, alloc->k_view(1).subview(0u, cross_tokens * vpt));
        const auto v1 = read_back(stream, alloc->v_view(1).subview(0u, cross_tokens * vpt));
        check_exact(k0, pattern(cross_tokens * vpt, 0.25f), "layer 0 K full span across pages");
        check_exact(v0, pattern(cross_tokens * vpt, 0.125f), "layer 0 V full span across pages");
        check_exact(k1, pattern(cross_tokens * vpt, 4.0f), "layer 1 K full span across pages");
        check_exact(v1, pattern(cross_tokens * vpt, -0.0625f), "layer 1 V full span across pages");
    }

    // Commit the full configured capacity and confirm further growth keeps
    // the already-written data stable.
    check(alloc->ensure_capacity(stream, max_context).ok(), "commit full capacity");
    check(alloc->committed_tokens() == max_context &&
              alloc->committed_pages_per_region() == alloc->pages_per_region() &&
              alloc->mapped_heap_count() == expected_heaps(max_context) &&
              alloc->committed_bytes() ==
                  static_cast<uint64_t>(alloc->mapped_heap_count()) * alloc->page_size_bytes(),
          "full-capacity residency accounting");
    {
        const auto k1 = read_back(stream, alloc->k_view(1).subview(0u, cross_tokens * vpt));
        check_exact(k1, pattern(cross_tokens * vpt, 4.0f), "data stable at full capacity");
    }
    // Growth from one page to ALL pages must not disturb either page: write a
    // page-1 marker and re-read it after the last growth step above.
    {
        const uint32_t marker_elems = std::min<uint32_t>(32u, vpt * 2u);
        const uint32_t marker_base = static_cast<uint32_t>(page_elems);
        dispatch_window(window_shader, stream, alloc->k_view(1),
                        marker_base, marker_elems, 4096.0f);
        stream.synchronize();
        const auto got = read_back(stream, alloc->k_view(1).subview(marker_base, marker_elems));
        check_exact(got, window_pattern(4096.0f, marker_elems),
                    "page-crossing growth preserved page-1 bytes");
    }

    // ------------------------------------------------------------------
    // Stage 3: NO-ALIAS PROBE. For every region and every committed page,
    // write a (region, page)-distinct pattern into the FIRST tile of the
    // page and another into its LAST tile, then read both back. A page
    // mapped twice out of one heap base (the DX RangeStart trap) makes two
    // windows land on the same bytes and the very first comparison fails.
    // Windows are 32 elements so the probe stays cheap while still touching
    // both page ends.
    // ------------------------------------------------------------------
    {
        const uint32_t committed_pages = alloc->committed_pages_per_region();
        const uint32_t pages_per_region = alloc->pages_per_region();
        const uint32_t window = 32u;
        const uint64_t logical_elems = static_cast<uint64_t>(max_context) * vpt;
        // First element of page p, and the last element of page p that still
        // lives inside the region's logical view (the final page of a region may
        // be partially used; its tail is mapped but not addressable through the
        // dense view, so the "last tile" probe is clamped into the view).
        const auto page_first = [&](uint32_t p) {
            return static_cast<uint32_t>(static_cast<uint64_t>(p) * page_elems);
        };
        const auto page_last_window = [&](uint32_t p) {
            const uint64_t end = std::min<uint64_t>(static_cast<uint64_t>(p + 1u) * page_elems,
                                                    logical_elems);
            return static_cast<uint32_t>(end - window);
        };
        const auto tag_for = [&](uint32_t r, uint32_t p, uint32_t end) {
            // Unique per (region, page, end) and small enough to stay exact in
            // FP32; the +i inside a window can never collide across tags.
            return static_cast<float>(1000u * (r * pages_per_region + p) + (end == 0u ? 7u : 507u));
        };
        bool windows_written = true;
        for (uint32_t r = 0u; r < region_count; ++r) {
            // Region order is [K0,V0,K1,V1,...]; resolving the view through the
            // public accessors keeps the probe backend-agnostic.
            const auto view = (r % 2u == 0u) ? alloc->k_view(r / 2u) : alloc->v_view(r / 2u);
            for (uint32_t p = 0u; p < committed_pages; ++p) {
                windows_written = windows_written && page_first(p) + window <= logical_elems &&
                                  page_last_window(p) >= page_first(p) + window;
                dispatch_window(window_shader, stream, view, page_first(p), window,
                                tag_for(r, p, 0u));
                dispatch_window(window_shader, stream, view, page_last_window(p), window,
                                tag_for(r, p, 1u));
            }
        }
        stream.synchronize();
        check(windows_written, "no-alias probe windows fit inside every page");
        bool first_ok = true, last_ok = true;
        for (uint32_t r = 0u; r < region_count; ++r) {
            const auto view = (r % 2u == 0u) ? alloc->k_view(r / 2u) : alloc->v_view(r / 2u);
            for (uint32_t p = 0u; p < committed_pages; ++p) {
                const auto got_first = read_back(stream, view.subview(page_first(p), window));
                const auto got_last = read_back(stream, view.subview(page_last_window(p), window));
                first_ok = first_ok && std::equal(got_first.begin(), got_first.end(),
                                                  window_pattern(tag_for(r, p, 0u), window).begin());
                last_ok = last_ok && std::equal(got_last.begin(), got_last.end(),
                                                window_pattern(tag_for(r, p, 1u), window).begin());
            }
        }
        check(first_ok, "no-alias probe: first-tile window of every (region, page) is distinct");
        check(last_ok, "no-alias probe: last-tile window of every (region, page) is distinct");

        // ------------------------------------------------------------------
        // Stage 4: one contiguous write straddling a page boundary inside a
        // region reads back intact (the write itself is a single dispatch over
        // a subview that spans two pages).
        // ------------------------------------------------------------------
        const uint32_t straddle_tokens = 4u;
        const uint32_t straddle_start_token = tokens_per_page - 1u;
        const uint64_t straddle_start =
            static_cast<uint64_t>(straddle_start_token) * vpt;
        const bool fits = straddle_start + straddle_tokens * vpt <= logical_elems;
        if (fits) {
            const auto start = static_cast<uint32_t>(straddle_start);
            const auto count = straddle_tokens * vpt;
            // The span starts one token before the page boundary, so it crosses
            // page 0 -> page 1 inside the region by construction.
            const bool crosses = straddle_start < page_elems &&
                                 straddle_start + count > page_elems;
            dispatch_write(shader, stream, alloc->k_view(2).subview(start, count),
                           alloc->v_view(2).subview(start, count), count, 0.5f, -1.0f);
            stream.synchronize();
            const auto got_k = read_back(stream, alloc->k_view(2).subview(start, count));
            const auto got_v = read_back(stream, alloc->v_view(2).subview(start, count));
            check(crosses && std::equal(got_k.begin(), got_k.end(), pattern(count, 0.5f).begin()),
                  "page-straddling write spans a boundary and reads back intact (K)");
            check(std::equal(got_v.begin(), got_v.end(), pattern(count, -1.0f).begin()),
                  "page-straddling write reads back intact (V)");
        } else {
            // Cannot happen for the geometry above (region >= 2 pages), but never
            // silently skip a check.
            check(false, "page-straddling write geometry", "straddle span exceeds the region view");
        }
    }

    // Reset after a dispatched write (the write is still in flight when
    // reset is entered; reset itself synchronizes before unmapping).
    dispatch_write(shader, stream, alloc->k_view(0), alloc->v_view(0), 16u, 8.0f, 8.0f);
    {
        auto st = alloc->reset(stream);
        check(st.ok(), "reset after dispatched work", st.message);
    }
    check(alloc->committed_bytes() == 0u && alloc->mapped_heap_count() == 0u &&
              alloc->committed_tokens() == 0u && alloc->committed_pages_per_region() == 0u,
          "reset drops all residency bookkeeping");
    check(alloc->valid() && alloc->reserved_bytes() % tile_bytes == 0u &&
              alloc->k_view(0).size() == max_context * vpt &&
              alloc->k_view(0).offset() == k0_offset_before,
          "buffer and views survive reset");

    // Remap and reuse: fresh page heaps back the same views; a new pattern reads
    // back exactly (content after reset is undefined until rewritten).
    check(alloc->ensure_capacity(stream, 4u).ok(), "remap after reset");
    check(alloc->mapped_heap_count() == expected_heaps(4u) &&
              alloc->committed_pages_per_region() == expected_pages(4u) &&
              alloc->committed_tokens() == 4u,
          "post-reset residency accounting");
    // Spec check 6: the same token count must remap the same page count and keep
    // every virtual address, then a larger count must add pages without moving
    // a single view.
    {
        const auto k_before = alloc->k_view(0).offset();
        const auto v_before = alloc->v_view(2).offset();
        const uint32_t pages_before = alloc->committed_pages_per_region();
        const bool same_again = alloc->ensure_capacity(stream, 4u).ok() &&
                                alloc->committed_pages_per_region() == pages_before &&
                                alloc->mapped_heap_count() == expected_heaps(4u);
        check(same_again && alloc->k_view(0).offset() == k_before &&
                  alloc->v_view(2).offset() == v_before,
              "re-ensure to the same token count keeps addresses and page count");
        const bool bigger = alloc->ensure_capacity(stream, cross_tokens).ok();
        check(bigger && alloc->committed_pages_per_region() == expected_pages(cross_tokens) &&
                  alloc->committed_pages_per_region() > pages_before &&
                  alloc->mapped_heap_count() == expected_heaps(cross_tokens) &&
                  alloc->k_view(0).offset() == k_before &&
                  alloc->v_view(2).offset() == v_before,
              "re-ensure to a larger token count remaps more pages at the same addresses");
    }
    dispatch_write(shader, stream, alloc->k_view(0), alloc->v_view(0),
                   4u * vpt, 0.25f, 0.125f);
    dispatch_write(shader, stream, alloc->k_view(2), alloc->v_view(2),
                   4u * vpt, -0.5f, 2.0f);
    stream.synchronize();
    {
        const auto k0 = read_back(stream, alloc->k_view(0).subview(0u, 4u * vpt));
        const auto v2 = read_back(stream, alloc->v_view(2).subview(0u, 4u * vpt));
        check_exact(k0, pattern(4u * vpt, 0.25f), "layer 0 K reused after reset");
        check_exact(v2, pattern(4u * vpt, 2.0f), "layer 2 V reused after reset");
    }

    // Deferred teardown coverage: dispatch without a synchronize and destroy
    // the allocator while the write may still be in flight; the destructor
    // must drain, unmap, and retire through its completion callback.
    dispatch_write(shader, stream, alloc->k_view(2), alloc->v_view(2), 4u * vpt, 1.0f, 1.0f);
    alloc = nullptr;
    stream.synchronize();
    check(true, "destructor after in-flight dispatch (main)");

    // ------------------------------------------------------------------
    // Stage 5: envelope guards and configuration equivalence.
    // ------------------------------------------------------------------
    // A page hint that is not a whole number of tiles is unusable and must fail
    // closed instead of being silently rounded (a caller-supplied residency
    // budget must never be reinterpreted).
    {
        QwenKvCacheAllocator bad;
        auto st = bad.create(device, stream, 16u, 8u, 1u,
                             static_cast<uint64_t>(tile_bytes) + 1u);
        check(!st.ok() && st.code == StatusCode::invalid_argument,
              "kv_page_bytes that is not a tile multiple fails closed");
        check(!bad.valid() && bad.heap_slot_count() == 0u,
              "rejected page hint leaves no allocator state");
    }
    // A virtual tile count above the uint32 sparse envelope is out_of_memory.
    // layer_count is derived from tiles_per_page so the envelope is exceeded on
    // every backend without allocating anything.
    {
        const uint64_t tiles_per_page = page_bytes / tile_bytes;
        const uint64_t overflow_layers = 0xFFFFFFFFull / (2ull * tiles_per_page) + 2ull;
        QwenKvCacheAllocator big;
        auto st = big.create(device, stream, 8u, 1u, static_cast<uint32_t>(overflow_layers));
        check(!st.ok() && st.code == StatusCode::out_of_memory && !big.valid(),
              "total-tile overflow fails closed with out_of_memory");
    }
    // Spec check 8: kv_page_bytes == tile_bytes (legacy one-tile pages where the
    // backend tile allows it -- the 2 MiB floor still applies on DX/VK) must be
    // functionally interchangeable with the default paged configuration: the
    // same token sequence produces byte-identical contents.
    {
        const uint32_t lvpt = 12u;
        const uint32_t llayers = 2u;
        const uint32_t ltokens = 16u;
        const uint32_t lmax = 64u;
        auto paged = luisa::make_unique<QwenKvCacheAllocator>();
        auto legacy = luisa::make_unique<QwenKvCacheAllocator>();
        auto st_paged = paged->create(device, stream, lmax, lvpt, llayers);
        auto st_legacy = legacy->create(device, stream, lmax, lvpt, llayers,
                                        static_cast<uint64_t>(tile_bytes));
        check(st_paged.ok() && st_legacy.ok(),
              "paged and one-tile-page configurations create",
              st_paged.ok() ? st_legacy.message : st_paged.message);
        check(legacy->page_size_bytes() ==
                  ceil_div(std::max(kMinPageBytes, static_cast<uint64_t>(tile_bytes)),
                           tile_bytes) *
                      tile_bytes,
              "one-tile-page hint still honors the 2 MiB page floor");
        if (st_paged.ok() && st_legacy.ok()) {
            const uint32_t n = ltokens * lvpt;
            const bool grown = paged->ensure_capacity(stream, ltokens).ok() &&
                               legacy->ensure_capacity(stream, ltokens).ok();
            check(grown, "one-tile-page equivalence growth",
                  grown ? luisa::string{} : luisa::string{"ensure_capacity failed"});
            dispatch_write(shader, stream, paged->k_view(0), paged->v_view(1),
                           n, 0.25f, -0.125f);
            dispatch_write(shader, stream, legacy->k_view(0), legacy->v_view(1),
                           n, 0.25f, -0.125f);
            stream.synchronize();
            const auto pk = read_back(stream, paged->k_view(0).subview(0u, n));
            const auto lk = read_back(stream, legacy->k_view(0).subview(0u, n));
            check(std::equal(pk.begin(), pk.end(), lk.begin()) &&
                      std::equal(pk.begin(), pk.end(), pattern(n, 0.25f).begin()),
                  "one-tile pages and paged config are byte-identical (K)");
            const auto pv = read_back(stream, paged->v_view(1).subview(0u, n));
            const auto lv = read_back(stream, legacy->v_view(1).subview(0u, n));
            check(std::equal(pv.begin(), pv.end(), lv.begin()) &&
                      std::equal(pv.begin(), pv.end(), pattern(n, -0.125f).begin()),
                  "one-tile pages and paged config are byte-identical (V)");
        } else {
            check(false, "one-tile-page equivalence", "create failed");
        }
        paged = nullptr;
        legacy = nullptr;
        stream.synchronize();
    }

    // ------------------------------------------------------------------
    // Stage 6: the REAL engine geometry (max_context 4096, 1024 values per
    // token, 8 full-attention layers = 16 regions) -- exactly the
    // configuration opt_plan/01-kv-cache-pages.md section 2.6 is about.
    // Residency is MEASURED, not assumed: a short session commits one page per
    // region (16 x 2 MiB = 32 MiB) instead of the old 7 tiles per region, and a
    // full 4096-token context needs 128 heaps instead of 4096.
    // ------------------------------------------------------------------
    {
        const uint32_t engine_vpt = 1024u;
        const uint32_t engine_layers = 8u;// ->16 regions (K and V per layer)
        const uint32_t engine_max = 4096u;
        const uint64_t region_logical = static_cast<uint64_t>(engine_max) * engine_vpt * sizeof(float);
        QwenKvCacheAllocator kv;
        auto st = kv.create(device, stream, engine_max, engine_vpt, engine_layers);
        check(st.ok(), "engine-geometry allocator create", st.message);
        if (st.ok()) {
            const uint64_t page = kv.page_size_bytes();
            const uint64_t regions = 2ull * engine_layers;
            uint64_t bytes[3] = {};
            uint32_t heaps[3] = {};
            const uint32_t tokens[3] = {128u, 512u, engine_max};
            bool grown = true;
            for (uint32_t i = 0u; i < 3u; ++i) {
                grown = grown && kv.ensure_capacity(stream, tokens[i]).ok();
                bytes[i] = kv.committed_bytes();
                heaps[i] = kv.mapped_heap_count();
            }
            // Page model expected residency, derived (never hardcoded).
            const auto expect_pages = [&](uint32_t t) {
                return static_cast<uint32_t>(
                    ceil_div(static_cast<uint64_t>(t) * engine_vpt * sizeof(float), page));
            };
            std::printf("engine geometry: page %u B, %u tiles/page, %u pages/region, "
                        "%u heap slots (maxMemoryAllocationCount 4096); committed "
                        "128 tokens = %llu B (%u heaps), 512 tokens = %llu B (%u heaps), "
                        "4096 tokens = %llu B (%u heaps)\n",
                        kv.page_size_bytes(), kv.tiles_per_page(), kv.pages_per_region(),
                        kv.heap_slot_count(),
                        static_cast<unsigned long long>(bytes[0]), heaps[0],
                        static_cast<unsigned long long>(bytes[1]), heaps[1],
                        static_cast<unsigned long long>(bytes[2]), heaps[2]);
            check(grown && kv.pages_per_region() == ceil_div(region_logical, page) &&
                      kv.heap_slot_count() == regions * kv.pages_per_region(),
                  "engine-geometry page count matches the spec formula");
            check(bytes[0] == regions * expect_pages(128u) * page &&
                      heaps[0] == regions * expect_pages(128u) &&
                      bytes[1] == regions * expect_pages(512u) * page &&
                      heaps[1] == regions * expect_pages(512u) &&
                      bytes[2] == regions * expect_pages(engine_max) * page &&
                      heaps[2] == kv.heap_slot_count(),
                  "engine-geometry committed bytes == heaps * page at 128/512/4096 tokens");
            check(kv.reset(stream).ok() && kv.committed_bytes() == 0u &&
                      kv.mapped_heap_count() == 0u,
                  "engine-geometry allocator reset");
        }
    }

    // ------------------------------------------------------------------
    // Stage 7: the long-context geometry of the plan's risk R3. At
    // max_context 262144 the default 2 MiB page still needs
    // 512 pages/region x 16 regions = 8192 heap slots, above the commonly
    // reported Vulkan maxMemoryAllocationCount of 4096. The allocator must NOT
    // clamp or fail: it reserves the 16 GiB VIRTUAL range (no heap is
    // committed) and warns with the kv_page_bytes value that would fit. Only
    // the geometry is asserted here; the warning itself goes to the log.
    // ------------------------------------------------------------------
    {
        const uint32_t long_max = 262144u;
        const uint32_t long_vpt = 1024u;
        const uint32_t long_layers = 8u;
        QwenKvCacheAllocator kv;
        auto st = kv.create(device, stream, long_max, long_vpt, long_layers);
        check(st.ok(), "long-context geometry create (reserves virtual only)", st.message);
        if (st.ok()) {
            const uint32_t page_slots = static_cast<uint32_t>(
                ceil_div(static_cast<uint64_t>(long_max) * long_vpt * sizeof(float),
                         kv.page_size_bytes()));
            check(kv.pages_per_region() == page_slots &&
                      kv.heap_slot_count() == 2ull * long_layers * page_slots &&
                      kv.heap_slot_count() > 4096u &&
                      kv.committed_bytes() == 0u && kv.mapped_heap_count() == 0u,
                  "long-context geometry exceeds 4096 heap slots without committing");
            std::printf("long-context geometry: max_context %u -> %u pages/region, "
                        "%u heap slots, %llu MiB virtual, 0 committed (R3 warning above "
                        "names the kv_page_bytes that would fit)\n",
                        long_max, kv.pages_per_region(), kv.heap_slot_count(),
                        static_cast<unsigned long long>(kv.reserved_bytes() / (1024ull * 1024ull)));
        }
    }

    // ------------------------------------------------------------------
    // Stage 8: BF16 + 64-position blocked layout (opt_plan/05-quantization.md
    // Q3). Same allocator, half the bytes per stored value, and a residency
    // set that is a per-head PAGE UNION rather than a prefix: each head owns a
    // contiguous block array, so the pages one region needs for N positions are
    // the pages intersecting every head's first ceil(N/64) blocks. Mapping a
    // linear prefix instead would leave a head's first block unmapped -- that
    // is exactly what these checks would catch.
    // ------------------------------------------------------------------
    {
        const uint32_t bf_kvh = 4u, bf_hd = 64u, bf_vpt = bf_kvh * bf_hd;
        const uint32_t bf_layers = 2u;
        const uint32_t bf_max = 19250u;// ragged: 301 blocks, last block padded by 14
        QwenKvLayoutDesc desc{};
        check(qwen_kv_layout_desc(QwenKvDType::bf16, QwenKvLayout::blocked64, bf_max, bf_kvh, bf_hd, desc).ok(),
              "bf16 allocator: contract desc");
        QwenKvCacheAllocator kv;
        auto st = kv.create(device, stream, bf_max, bf_vpt, bf_layers, 0u,
                            QwenKvDType::bf16, bf_kvh, QwenKvLayout::blocked64);
        check(st.ok(), "bf16 allocator create", st.message);
        if (!st.ok()) { return summarize("qwen kv cache", argv[1]); }
        check(kv.kv_dtype() == QwenKvDType::bf16 && kv.kv_layout() == QwenKvLayout::blocked64 &&
                  kv.elements_per_word() == 2u && kv.kv_heads() == bf_kvh &&
                  kv.head_dim() == bf_hd && kv.position_blocks() == desc.position_blocks,
              "bf16 allocator: dtype/layout/head geometry accessors");
        const uint64_t bf_regions = 2ull * bf_layers;
        check(kv.region_words() == desc.region_words &&
                  kv.logical_bytes() == bf_regions * desc.region_words * sizeof(float) &&
                  kv.payload_bytes() == bf_regions * desc.payload_bytes() &&
                  kv.payload_bytes() == bf_regions * static_cast<uint64_t>(bf_max) * bf_vpt * 2ull,
              "bf16 allocator: region words / logical / payload bytes follow the contract");
        check(kv.pages_per_region() ==
                  ceil_div(desc.region_words * sizeof(float), kv.page_size_bytes()) &&
                  kv.committed_bytes() == 0u && kv.mapped_heap_count() == 0u,
              "bf16 allocator: pages/region from the byte geometry, zero residency after create");
        {
            bool views_ok = true;
            luisa::vector<uint64_t> bases;
            const uint64_t page_elems = kv.page_size_bytes() / sizeof(float);
            for (uint32_t l = 0u; l < bf_layers; ++l) {
                const auto k = kv.k_view(l);
                const auto v = kv.v_view(l);
                const auto kw = kv.k_word_view(l);
                views_ok = views_ok && k.size() == desc.region_words &&
                           v.size() == desc.region_words && kw.size() == desc.region_words &&
                           kw.offset() == k.offset() &&
                           k.offset() % page_elems == 0u && v.offset() % page_elems == 0u;
                bases.emplace_back(k.offset());
                bases.emplace_back(v.offset());
            }
            std::sort(bases.begin(), bases.end());
            check(views_ok && std::adjacent_find(bases.begin(), bases.end()) == bases.end(),
                  "bf16 allocator: views are region_words long, page-aligned and distinct");
        }
        // Spec residency: the union of the per-head page runs (independent of the
        // allocator's implementation).
        const auto spec_pages = [&](uint32_t tokens) {
            const uint64_t page_words = kv.page_size_bytes() / sizeof(float);
            luisa::vector<uint8_t> flags(kv.pages_per_region(), 0u);
            const uint64_t blocks = std::min<uint64_t>((tokens + 63u) / 64u, desc.position_blocks);
            const uint64_t run_words = blocks * desc.block_codes() / desc.elements_per_word();
            for (uint32_t h = 0u; h < bf_kvh; ++h) {
                const uint64_t base = static_cast<uint64_t>(h) * desc.head_stride_codes /
                                      desc.elements_per_word();
                const uint64_t first = base / page_words;
                const uint64_t last = (base + run_words - 1u) / page_words;
                for (uint64_t pg = first; pg <= last && pg < kv.pages_per_region(); ++pg) {
                    flags[static_cast<size_t>(pg)] = 1u;
                }
            }
            uint32_t n = 0u;
            for (const auto f : flags) { n += f != 0u ? 1u : 0u; }
            return n;
        };
        check(kv.min_commit_pages() == spec_pages(1u) && kv.min_commit_pages() > 1u,
              "bf16 allocator: one token commits a multi-page head union (blocked granularity)");
        // Ragged 64-position boundaries and page-straddling growth. Every step
        // re-derives the expected page count from the contract.
        {
            const uint32_t steps[] = {1u, 64u, 65u, 5000u, 17000u, bf_max};
            bool growth_ok = true;
            uint64_t prev = 0u;
            luisa::string detail;
            for (const auto t : steps) {
                growth_ok = growth_ok && kv.ensure_capacity(stream, t).ok();
                const bool step_ok =
                    kv.committed_tokens() == t &&
                    kv.committed_pages_per_region() == spec_pages(t) &&
                    kv.mapped_heap_count() == 2u * bf_layers * kv.committed_pages_per_region() &&
                    kv.committed_bytes() ==
                        static_cast<uint64_t>(kv.mapped_heap_count()) * kv.page_size_bytes() &&
                    kv.committed_bytes() >= prev;
                if (!step_ok) {
                    detail = luisa::string{"at "} + std::to_string(t).c_str() +
                             " tokens: pages " + std::to_string(kv.committed_pages_per_region()).c_str() +
                             ", spec " + std::to_string(spec_pages(t)).c_str() +
                             ", heaps " + std::to_string(kv.mapped_heap_count()).c_str();
                }
                growth_ok = growth_ok && step_ok;
                prev = kv.committed_bytes();
            }
            check(growth_ok, "bf16 allocator: growth commits exactly the intersecting per-head page set", detail);
            std::printf("bf16 geometry: page %u B, %u pages/region, first-token %u page(s)/region, "
                        "full capacity %u pages/region x %llu regions = %u heaps, payload %.2f MiB\n",
                        kv.page_size_bytes(), kv.pages_per_region(), kv.min_commit_pages(),
                        kv.committed_pages_per_region(),
                        static_cast<unsigned long long>(bf_regions), kv.mapped_heap_count(),
                        static_cast<double>(kv.payload_bytes()) / 1048576.0);
        }
        // Per-head no-alias probe: a head-and-block distinct marker at every
        // head's block 0 and block 1 start. Two heads (or two blocks) resolving
        // to one heap/mapping collide on the first compared word.
        const auto head_block_words = static_cast<uint32_t>(desc.block_codes() / desc.elements_per_word());
        const auto k_words = kv.k_word_view(0u).as<float>();
        {
            bool written = true;
            for (uint32_t h = 0u; h < bf_kvh; ++h) {
                for (uint32_t b = 0u; b < 2u; ++b) {
                    const uint32_t base = h * static_cast<uint32_t>(desc.head_stride_codes /
                                                                   desc.elements_per_word()) +
                                          b * head_block_words;
                    dispatch_window(window_shader, stream, k_words, base, 8u,
                                    static_cast<float>(1000u * h + 16u * b));
                    written = written && base + 8u <= k_words.size();
                }
            }
            stream.synchronize();
            bool ok = written;
            for (uint32_t h = 0u; h < bf_kvh && ok; ++h) {
                for (uint32_t b = 0u; b < 2u && ok; ++b) {
                    const uint32_t base = h * static_cast<uint32_t>(desc.head_stride_codes /
                                                                   desc.elements_per_word()) +
                                          b * head_block_words;
                    const auto got = read_back(stream, k_words.subview(base, 8u));
                    const auto want = window_pattern(static_cast<float>(1000u * h + 16u * b), 8u);
                    ok = std::equal(got.begin(), got.end(), want.begin());
                }
            }
            check(ok, "bf16 allocator: per-head block markers are distinct (no aliasing, one heap per page)");
        }
        // Data preservation across a growth step, then reset/remap.
        {
            const uint32_t marker_base = head_block_words * 2u + 4u;
            dispatch_window(window_shader, stream, k_words, marker_base, 8u, 4096.0f);
            stream.synchronize();
            const auto before = read_back(stream, k_words.subview(marker_base, 8u));
            check(kv.reset(stream).ok() && kv.committed_pages_per_region() == 0u &&
                      kv.mapped_heap_count() == 0u && kv.k_view(0u).offset() == k_words.offset(),
                  "bf16 allocator: reset drops residency and keeps the virtual addresses");
            check(kv.ensure_capacity(stream, bf_max).ok() &&
                      kv.committed_pages_per_region() == kv.pages_per_region() &&
                      kv.mapped_heap_count() == 2u * bf_layers * kv.pages_per_region(),
                  "bf16 allocator: remap after reset restores the full page set");
            dispatch_window(window_shader, stream, k_words, marker_base, 8u, 4096.0f);
            stream.synchronize();
            const auto after = read_back(stream, k_words.subview(marker_base, 8u));
            check(std::equal(before.begin(), before.end(), after.begin()),
                  "bf16 allocator: the same words address the same data after reset/remap");
        }
        // FP32 A/B at the SAME geometry: identical payload halved and one page
        // per region for a single token instead of one per head.
        {
            QwenKvCacheAllocator fp32;
            auto fst = fp32.create(device, stream, bf_max, bf_vpt, bf_layers, 0u,
                                   QwenKvDType::f32, bf_kvh, QwenKvLayout::position_major);
            check(fst.ok(), "fp32 A/B allocator create", fst.message);
            if (fst.ok()) {
            check(fp32.payload_bytes() == 2u * kv.payload_bytes() &&
                      fp32.region_words() == static_cast<uint64_t>(bf_max) * bf_vpt &&
                      fp32.logical_bytes() > kv.logical_bytes() &&
                      fp32.logical_bytes() <= 2u * kv.logical_bytes(),
                  "fp32 A/B: exact 2x payload; blocked64 tail padding is separate and bounded");
                check(fp32.min_commit_pages() == 1u &&
                          fp32.ensure_capacity(stream, 1u).ok() &&
                          fp32.committed_pages_per_region() == 1u &&
                          kv.min_commit_pages() == spec_pages(1u) &&
                          kv.min_commit_pages() > fp32.min_commit_pages(),
                      "fp32 A/B: a token needs one page per region at fp32 vs the head union at bf16");
                check(fp32.elements_per_word() == 1u && fp32.position_blocks() == 0u,
                      "fp32 A/B: one code per word and no block addressing");
            }
        }

// ------------------------------------------------------------------
// fp8_e4m3 + blocked64 DEVICE stage (opt_plan/05 Q3, FP8 step): the
// dtype the online-scale append kernel writes. The allocator must
// own the dense per-region scale arrays (outside the sparse mapping,
// so a scale read can never be unmapped), keep them zero until a
// block is written, clear them on reset, and price the cache
// honestly: a quarter of the FP32 payload in codes, plus the scales.
// ------------------------------------------------------------------
{
    const uint32_t f8_kvh = bf_kvh, f8_hd = bf_hd, f8_vpt = bf_vpt;
    const uint32_t f8_max = bf_max;
    QwenKvLayoutDesc f8d{};
    check(qwen_kv_layout_desc(QwenKvDType::fp8_e4m3, QwenKvLayout::blocked64,
                              f8_max, f8_kvh, f8_hd, f8d).ok(),
          "fp8 allocator: contract desc");
    QwenKvCacheAllocator f8;
    auto st8 = f8.create(device, stream, f8_max, f8_vpt, bf_layers, 0u,
                         QwenKvDType::fp8_e4m3, f8_kvh, QwenKvLayout::blocked64);
    check(st8.ok(), "fp8 allocator create", st8.message);
    if (st8.ok()) {
        check(f8.kv_dtype() == QwenKvDType::fp8_e4m3 &&
                  f8.kv_layout() == QwenKvLayout::blocked64 &&
                  f8.elements_per_word() == 4u && f8.kv_heads() == f8_kvh &&
                  f8.head_dim() == f8_hd && f8.position_blocks() == f8d.position_blocks &&
                  f8.scales_per_region() == f8d.scale_count,
              "fp8 allocator: dtype/layout/head/scale geometry accessors");
        // Payload: exactly a quarter of the FP32 cache at the same geometry, and
        // the dense scales are reported separately AND inside logical_bytes (they
        // are real memory, so hiding them would flatter the capacity number).
        check(f8.payload_bytes() * 4ull ==
                      static_cast<uint64_t>(bf_regions) * static_cast<uint64_t>(f8_max) * f8_vpt * 4ull &&
                  f8.payload_bytes() * 2ull == kv.payload_bytes(),
              "fp8 allocator: payload is a quarter of fp32 (half of bf16)");
        check(f8.scale_bytes() ==
                      static_cast<uint64_t>(bf_regions) * f8d.scale_bytes() &&
                  f8.scale_bytes() == bf_regions * f8_kvh * f8d.position_blocks * sizeof(float) &&
                  f8.logical_bytes() ==
                      bf_regions * f8d.region_words * sizeof(float) + f8.scale_bytes(),
              "fp8 allocator: dense scales are priced inside logical_bytes");
        check(f8.region_words() == f8d.region_words &&
                  f8.region_words() * 4u == f8d.region_codes && f8.region_words() * 4u >
                      static_cast<uint64_t>(bf_max) * f8_vpt * 1ull,
              "fp8 allocator: one region is region_words quads (padding included)");
        check(f8.pages_per_region() == kv.pages_per_region() / 2u + (kv.pages_per_region() % 2u) &&
                  f8.committed_bytes() == 0u && f8.mapped_heap_count() == 0u,
              "fp8 allocator: half the pages of bf16 and no residency after create");
        // Scale views: one dense buffer per (layer, K|V), scale_count long, stable
        // across growth and reset, and never inside the sparse region.
        const auto ksc0 = f8.k_scale_view(0u);
        const auto vsc0 = f8.v_scale_view(0u);
        const auto kv0 = f8.k_view(0u);
        check(ksc0.size() == f8d.scale_count && vsc0.size() == f8d.scale_count &&
                  ksc0.offset() == 0u && vsc0.offset() == 0u &&
                  ksc0.handle() != kv0.handle(),
              "fp8 allocator: scale views are dense, scale_count long and outside the sparse regions");
        const auto scales0 = read_back(stream, f8.k_scale_view(1u));
        bool zero = true;
        for (const auto s : scales0) { zero = zero && s == 0.0f; }
        check(zero, "fp8 allocator: every scale starts at 0 (the never-written sentinel)");
        // Write a scale pattern through the same view an engine binds, then check
        // growth and reset keep the allocation (and its contents) stable.
        {
            luisa::vector<float> pattern(scales0.size());
            for (size_t i = 0; i < pattern.size(); ++i) {
                pattern[i] = static_cast<float>(i) * 0.5f;
            }
            stream << f8.k_scale_view(0u).copy_from(luisa::span<const float>{pattern})
                   << synchronize();
            const auto got = read_back(stream, f8.k_scale_view(0u));
            check(got == pattern, "fp8 allocator: the scale view accepts a host upload");
            check(f8.ensure_capacity(stream, 65u).ok() &&
                      f8.k_scale_view(0u).handle() == ksc0.handle() &&
                      f8.k_scale_view(0u).size() == ksc0.size() &&
                      read_back(stream, f8.k_scale_view(0u)) == pattern,
                  "fp8 allocator: growth does not move or clear the scales");
            check(f8.reset(stream).ok() && read_back(stream, f8.k_scale_view(0u)) ==
                                                 luisa::vector<float>(pattern.size(), 0.0f),
                  "fp8 allocator: reset clears every scale back to the sentinel");
        }
        // Residency: the same per-head page union as bf16 (the codes are narrower,
        // not the blocks), and the K/V region words are addressable through the
        // word view the append kernel binds.
        {
            // Spec residency of the fp8 blocked region: the union of the per-head
            // runs, ONE PAGE COVERING TWICE THE POSITIONS of the bf16 cache because a
            // code is one byte (so the union is smaller, never larger).
            const uint64_t page_words8 = f8.page_size_bytes() / sizeof(float);
            luisa::vector<uint8_t> flags(f8.pages_per_region(), 0u);
            const uint64_t words_per_head = f8d.head_stride_codes / 4u;
            for (uint32_t h = 0u; h < f8_kvh; ++h) {
                const uint64_t first = (h * words_per_head) / page_words8;
                const uint64_t last = (h * words_per_head + (64ull * f8_hd) / 4u - 1u) / page_words8;
                for (uint64_t pg = first; pg <= last && pg < f8.pages_per_region(); ++pg) {
                    flags[static_cast<size_t>(pg)] = 1u;
                }
            }
            uint32_t spec8 = 0u;
            for (const auto f : flags) { spec8 += f != 0u ? 1u : 0u; }
            check(f8.min_commit_pages() == spec8 && spec8 >= 1u &&
                      spec8 <= kv.min_commit_pages() &&
                      f8.ensure_capacity(stream, 1u).ok() &&
                      f8.committed_pages_per_region() == spec8,
                  "fp8 allocator: one token commits the per-head page union of the quarter-size codes");
        }
        check(f8.k_word_view(0u).size() == f8d.region_words &&
                  f8.v_word_view(0u).size() == f8d.region_words,
              "fp8 allocator: the word view is exactly region_words quads long");
        // Data preservation across growth, exactly like the bf16 stage.
        {
            const auto words0 = f8.k_word_view(0u);
            const uint32_t base = 3u;
            luisa::vector<uint32_t> marker{0x7E7E7E7Eu, 0x01020304u, 0xFFFFFFFFu, 0x00000000u};
            stream << words0.subview(base, marker.size())
                          .copy_from(luisa::span<const uint32_t>{marker})
                   << synchronize();
            check(f8.ensure_capacity(stream, 4096u).ok(), "fp8 allocator: growth to 4096");
            luisa::vector<uint32_t> back(marker.size());
            stream << f8.k_word_view(0u).subview(base, marker.size())
                          .copy_to(luisa::span<uint32_t>{back})
                   << synchronize();
            check(back == marker, "fp8 allocator: growth preserves the fp8 code words");
            check(f8.reset(stream).ok() && f8.committed_pages_per_region() == 0u &&
                      f8.mapped_heap_count() == 0u &&
                      f8.k_word_view(0u).size() == f8d.region_words &&
                      f8.k_scale_view(0u).size() == f8d.scale_count,
                  "fp8 allocator: reset keeps the allocator usable (views stable)");
            check(f8.ensure_capacity(stream, f8_max).ok() &&
                      f8.committed_pages_per_region() == f8.pages_per_region(),
                  "fp8 allocator: remap after reset covers the full capacity");
        }
        std::printf("fp8 geometry: page %u B, %u pages/region (bf16 %u), %u scales/region, "
                    "scale bytes %.1f KiB, payload %.2f MiB (fp32 %.2f MiB), first-token %u page(s)\n",
                    f8.page_size_bytes(), f8.pages_per_region(), kv.pages_per_region(),
                    f8.scales_per_region(), static_cast<double>(f8.scale_bytes()) / 1024.0,
                    static_cast<double>(f8.payload_bytes()) / 1048576.0,
                    static_cast<double>(f8.payload_bytes() * 4ull) / 1048576.0,
                    f8.min_commit_pages());
    }
}

    }
    return summarize("qwen kv cache", argv[1]);
}
