// Qwen3.5-4B full-attention KV cache -- dtype/layout contract (see the header).
// Host-only: geometry validation, the sparse residency extent helpers and the
// dense<->region converters the CPU oracles and the tests use.

#include "qwen/kv_cache_layout.h"

#include <algorithm>
#include <cmath>

namespace luisa::qwen {

namespace {
// Status carries luisa::string, so every message is materialized here (a
// string_view does not convert implicitly).
[[nodiscard]] Status err_invalid(const char *message, const char *field) noexcept {
    return Status::error(StatusCode::invalid_argument, luisa::string{message}, luisa::string{field});
}
[[nodiscard]] Status err_overflow(const char *message) noexcept {
    return Status::error(StatusCode::out_of_memory, luisa::string{message});
}
}// namespace

Status qwen_kv_layout_desc(QwenKvDType dtype, QwenKvLayout layout,
                           uint32_t max_context, uint32_t kv_heads,
                           uint32_t head_dim, QwenKvLayoutDesc &out) noexcept {
    if (max_context == 0u) {
        return err_invalid("KV layout requires max_context >= 1", "max_context");
    }
    if (kv_heads == 0u) {
        return err_invalid("KV layout requires kv_heads >= 1", "kv_heads");
    }
    if (head_dim == 0u) {
        return err_invalid("KV layout requires head_dim >= 1", "head_dim");
    }
    // A BF16 code pair lives in ONE 32-bit word, and every term of the index
    // arithmetic is a multiple of head_dim, so an even head_dim guarantees a pair
    // never crosses a position row. The same holds for an FP8 quad: it needs
    // head_dim % 4 == 0. That keeps the store side a single-word write (no
    // read-modify-write, no cross-thread race) on every layout.
    if (dtype == QwenKvDType::bf16 && (head_dim & 1u) != 0u) {
        return err_invalid("BF16 KV codes are packed two per word, so head_dim must be even",
                           "head_dim");
    }
    if (dtype == QwenKvDType::fp8_e4m3 && head_dim % 4u != 0u) {
        return err_invalid("FP8 E4M3 KV codes are packed four per word, so head_dim must be "
                           "a multiple of 4",
                           "head_dim");
    }
    // The scaled dtype derives its scale from the POSITION BLOCK, so it is only
    // representable in the blocked layout. Refusing here (rather than ignoring the
    // scale) is what keeps a position-major fp8 region from ever being built.
    if (qwen_kv_scaled(dtype) && layout != QwenKvLayout::blocked64) {
        return err_invalid("FP8 E4M3 KV keeps one scale per (kv_head, position block), so it "
                           "requires the blocked64 layout",
                           "layout");
    }
    QwenKvLayoutDesc desc{};
    desc.dtype = dtype;
    desc.layout = layout;
    desc.max_context = max_context;
    desc.kv_heads = kv_heads;
    desc.head_dim = head_dim;
    uint64_t codes_per_token = 0;
    if (!checked_mul(kv_heads, head_dim, codes_per_token)) {
        return err_overflow("KV layout: kv_heads * head_dim overflows");
    }
    if (layout == QwenKvLayout::position_major) {
        desc.position_blocks = 0u;
        desc.head_stride_codes = 0u;
        if (!checked_mul(max_context, codes_per_token, desc.region_codes)) {
            return err_overflow("KV layout: max_context * values_per_token overflows");
        }
    } else {
        const uint64_t blocks =
            (static_cast<uint64_t>(max_context) + qwen_kv_position_block - 1u) / qwen_kv_position_block;
        const uint64_t band_codes = static_cast<uint64_t>(qwen_kv_position_block) * head_dim;
        uint64_t head_codes = 0;
        if (!checked_mul(blocks, band_codes, head_codes)) {
            return err_overflow("KV layout: per-head blocked code count overflows");
        }
        if (!checked_mul(head_codes, kv_heads, desc.region_codes)) {
            return err_overflow("KV layout: blocked region code count overflows");
        }
        if (blocks > 0xFFFFFFFFull) {
            return err_invalid("KV layout: position block count exceeds uint32", "position_blocks");
        }
        desc.position_blocks = static_cast<uint32_t>(blocks);
        desc.head_stride_codes = head_codes;
    }
    if (desc.region_codes == 0u) {
        return err_overflow("KV layout: empty region");
    }
    if (qwen_kv_scaled(dtype)) {
        // Dense [kv_heads][position_blocks] FP32 scales. Both the element count and
        // the byte count are checked: the index arithmetic of the device kernels is
        // uint32, so a scale array that does not fit is unaddressable there.
        uint64_t scale_count = 0;
        if (!checked_mul(static_cast<uint64_t>(kv_heads), desc.position_blocks, scale_count) ||
            scale_count > 0xFFFFFFFFull / sizeof(float)) {
            return err_overflow("KV layout: fp8 scale array overflows");
        }
        desc.scale_count = static_cast<uint32_t>(scale_count);
    }
    if (desc.region_codes % qwen_kv_elements_per_word(dtype) != 0u) {
        // Cannot happen for an even head_dim in the blocked layout and never for
        // f32; keep the guard so a future layout change fails closed here rather
        // than truncating a trailing half word.
        return err_invalid("KV layout: region code count is not a whole number of words",
                           "region_words");
    }
    desc.region_words = desc.region_codes / qwen_kv_elements_per_word(dtype);
    out = desc;
    return Status::success();
}

Status qwen_kv_native_desc(QwenKvDType dtype, uint32_t max_context,
                           uint32_t kv_heads, uint32_t head_dim,
                           QwenKvLayoutDesc &out) noexcept {
    return qwen_kv_layout_desc(dtype, qwen_kv_native_layout(dtype), max_context,
                               kv_heads, head_dim, out);
}

Status qwen_kv_region_to_dense(const QwenKvLayoutDesc &desc,
                               luisa::span<const uint32_t> words,
                               luisa::span<const float> scales,
                               uint32_t tokens,
                               luisa::span<float> dense) noexcept {
    if (words.size() < desc.region_words) {
        return err_invalid("KV region word span is shorter than the configured region",
                           "words");
    }
    if (tokens > desc.max_context) {
        return err_invalid("KV dequantization covers more positions than max_context", "tokens");
    }
    if (desc.scaled() && scales.size() < desc.scale_count) {
        return err_invalid("a scaled (fp8_e4m3) KV region needs desc.scale_count scales to "
                           "dequantize; refusing to read codes without their block scale",
                           "scales");
    }
    const uint64_t codes = static_cast<uint64_t>(tokens) * desc.legacy_codes_per_token();
    if (dense.size() < codes) {
        return err_invalid("dense KV span is shorter than tokens * kv_heads * head_dim", "dense");
    }
    for (uint32_t head = 0u; head < desc.kv_heads; ++head) {
        for (uint32_t pos = 0u; pos < tokens; ++pos) {
            const auto block_scale = desc.scaled()
                                         ? scales[desc.scale_index(head, pos)]
                                         : 1.0f;
            for (uint32_t c = 0u; c < desc.head_dim; ++c) {
                const auto code = desc.code_index(head, pos, c);
                const auto value = qwen_kv_read_word(desc, words, code, block_scale);
                const auto at = (static_cast<uint64_t>(pos) * desc.legacy_codes_per_token()) +
                                static_cast<uint64_t>(head) * desc.head_dim + c;
                dense[at] = value;
            }
        }
    }
    return Status::success();
}

Status qwen_kv_region_to_dense(const QwenKvLayoutDesc &desc,
                               luisa::span<const uint32_t> words,
                               uint32_t tokens,
                               luisa::span<float> dense) noexcept {
    return qwen_kv_region_to_dense(desc, words, luisa::span<const float>{}, tokens, dense);
}

Status qwen_kv_region_from_dense(const QwenKvLayoutDesc &desc,
                                 luisa::span<float> scales,
                                 luisa::span<const float> dense,
                                 uint32_t tokens,
                                 luisa::span<uint32_t> words) noexcept {
    if (words.size() < desc.region_words) {
        return err_invalid("KV region word span is shorter than the configured region",
                           "words");
    }
    if (tokens > desc.max_context) {
        return err_invalid("KV layout covers more positions than max_context", "tokens");
    }
    if (desc.scaled() && scales.size() < desc.scale_count) {
        return err_invalid("a scaled (fp8_e4m3) KV region needs desc.scale_count scales",
                           "scales");
    }
    const uint64_t codes = static_cast<uint64_t>(tokens) * desc.legacy_codes_per_token();
    if (dense.size() < codes) {
        return err_invalid("dense KV span is shorter than tokens * kv_heads * head_dim", "dense");
    }
    if (desc.scaled()) {
        // Whole blocks: the scale is derived over the block's own valid values and
        // every code is encoded against it. Written through the same reference the
        // device append kernel mirrors, so a test that compares the two compares
        // exactly one implementation of the rule.
        std::fill(words.begin(), words.end(), 0u);
        std::fill(scales.begin(), scales.end(), 0.0f);
        return qwen_kv_append_reference(desc, dense, 0u, tokens, scales, words);
    }
    // Whole words are rebuilt from their own codes, so the two codes of a BF16
    // word are written by the same iteration (no read-modify-write race, no
    // assumption about the caller's initial content).
    for (uint32_t head = 0u; head < desc.kv_heads; ++head) {
        for (uint32_t pos = 0u; pos < tokens; ++pos) {
            for (uint32_t c = 0u; c < desc.head_dim; c += desc.elements_per_word()) {
                const auto code = desc.code_index(head, pos, c);
                uint32_t word = 0u;
                for (uint32_t k = 0u; k < desc.elements_per_word(); ++k) {
                    const auto at = (static_cast<uint64_t>(pos) * desc.legacy_codes_per_token()) +
                                    static_cast<uint64_t>(head) * desc.head_dim + c + k;
                    word |= static_cast<uint32_t>(qwen_kv_code_from_f32(dense[at]))
                            << (16u * k);
                }
                words[desc.word_index(code)] = word;
            }
        }
    }
    return Status::success();
}

Status qwen_kv_region_from_dense(const QwenKvLayoutDesc &desc,
                                 luisa::span<const float> dense,
                                 uint32_t tokens,
                                 luisa::span<uint32_t> words) noexcept {
    return qwen_kv_region_from_dense(desc, luisa::span<float>{}, dense, tokens, words);
}

Status qwen_kv_append_reference(const QwenKvLayoutDesc &desc,
                                luisa::span<const float> src,
                                uint32_t pos0, uint32_t tokens,
                                luisa::span<float> scales,
                                luisa::span<uint32_t> words) noexcept {
    if (words.size() < desc.region_words) {
        return err_invalid("KV region word span is shorter than the configured region",
                           "words");
    }
    const uint64_t src_codes =
        static_cast<uint64_t>(tokens) * desc.legacy_codes_per_token();
    if (tokens != 0u && src.size() < src_codes) {
        return err_invalid("dense KV span is shorter than tokens * kv_heads * head_dim", "src");
    }
    if (tokens != 0u && static_cast<uint64_t>(pos0) + tokens > desc.max_context) {
        return err_invalid("KV append exceeds max_context", "pos0 + tokens");
    }
    if (desc.scaled() && scales.size() < desc.scale_count) {
        return err_invalid("a scaled (fp8_e4m3) KV append needs desc.scale_count scales",
                           "scales");
    }
    const auto stored_value = [&](uint32_t head, uint32_t pos, uint32_t c,
                                  float block_scale) noexcept {
        // The stored value of one code, dequantized with the scale it was WRITTEN
        // with (the caller passes it: after a grow the array already holds the new
        // one, and decoding with it would scale the old code twice).
        return qwen_kv_read_word(desc, words, desc.code_index(head, pos, c), block_scale);
    };
    for (uint32_t head = 0u; head < desc.kv_heads; ++head) {
        if (tokens == 0u) { continue; }
        const uint32_t first_pos = pos0;
        const uint32_t last_pos = pos0 + tokens - 1u;
        const uint32_t first_block = first_pos / qwen_kv_position_block;
        const uint32_t last_block = last_pos / qwen_kv_position_block;
        for (uint32_t block = first_block; block <= last_block; ++block) {
            const uint32_t block_start = block * qwen_kv_position_block;
            const uint32_t block_end = std::min<uint32_t>(block_start + qwen_kv_position_block,
                                                          desc.max_context);
            const uint32_t valid_begin = block_start;
            const uint32_t valid_end = std::min<uint32_t>(last_pos + 1u, block_end);
            const uint32_t new_begin = std::max<uint32_t>(first_pos, block_start);
            // 1. The maximum magnitude of the NEW rows only. The block's earlier
            //    maximum is carried by its scale (see step 2), which is why no
            //    FP32 shadow copy of the block is needed.
            float max_abs = 0.0f;
            for (uint32_t pos = new_begin; pos < valid_end; ++pos) {
                const auto row = (static_cast<uint64_t>(pos - pos0) *
                                  desc.legacy_codes_per_token()) +
                                 static_cast<uint64_t>(head) * desc.head_dim;
                for (uint32_t c = 0u; c < desc.head_dim; ++c) {
                        const auto v = src[row + c];
                        // Non-finite model data contributes one full-scale unit,
                        // never infinity: a non-finite-only block therefore gets
                        // scale 1 and preserves its inf/NaN codes instead of 0.
                        const auto magnitude = qwen_kv_counts_for_max(v)
                                                   ? std::fabs(v)
                                                   : static_cast<float>(kFp8E4M3Max);
                        max_abs = std::max(max_abs, magnitude);
                }
            }
            // 2. The scale the block MUST have: the running maximum over every value
            //    ever appended, i.e. max(old_scale, max_abs/448). The old scale
            //    already covers every stored code, so it is the floor -- a block
            //    whose new values stay below 448*old_scale keeps its scale AND its
            //    bytes, and a grow rewrites the whole valid prefix against the new
            //    scale (never leaving an old code paired with it).
            const float old_scale = desc.scaled() ? scales[desc.scale_index(head, block_start)] : 1.0f;
            const float grown = desc.scaled() ? qwen_kv_block_scale(max_abs) : 1.0f;
            const float new_scale = std::max(old_scale, grown);
            const bool grows = new_scale > old_scale;
            if (desc.scaled()) { scales[desc.scale_index(head, block_start)] = new_scale; }
            // 3. Rewrite. Growing the scale rewrites EVERY valid position of the
            //    block (earlier codes are re-encoded against the new scale);
            //    otherwise only the new rows are written, and each word is rebuilt
            //    by one iteration (a packed word is never shared between lanes).
            const uint32_t write_begin = grows ? valid_begin : new_begin;
            for (uint32_t pos = write_begin; pos < valid_end; ++pos) {
                const auto row = (static_cast<uint64_t>(pos - pos0) *
                                  desc.legacy_codes_per_token()) +
                                 static_cast<uint64_t>(head) * desc.head_dim;
                const bool is_new = pos >= new_begin;
                for (uint32_t c = 0u; c < desc.head_dim; c += desc.elements_per_word()) {
                    const auto code = desc.code_index(head, pos, c);
                    uint32_t word = 0u;
                    for (uint32_t k = 0u; k < desc.elements_per_word(); ++k) {
                        float value;
                        if (is_new) {
                            value = src[row + c + k];
                        } else {
                            value = stored_value(head, pos, c + k, old_scale);
                        }
                        uint32_t code_bits;
                        if (desc.dtype == QwenKvDType::fp8_e4m3) {
                            code_bits = qwen_kv_fp8_code_from_scaled(value, new_scale);
                        } else if (desc.dtype == QwenKvDType::bf16) {
                            code_bits = qwen_kv_code_from_f32(value);
                        } else {
                            code_bits = qwen_kv_f32_as_word(value);
                        }
                        word |= code_bits << (8u * qwen_kv_element_bytes(desc.dtype) * k);
                    }
                    words[desc.word_index(code)] = word;
                }
            }
        }
    }
    return Status::success();
}

}// namespace luisa::qwen
