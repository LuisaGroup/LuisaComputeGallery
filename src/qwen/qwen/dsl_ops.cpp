// Qwen3.5-4B ordinary DSL operators -- implementation.
// Device-free kernel factories, overflow-checked host planning/validation,
// independent double-accumulation CPU oracles, and a device-free self-test.
// The exact contracts/layouts are declared in dsl_ops.h.
#include "qwen/dsl_ops.h"

// Host-only Q8_0 block primitive and the storage-enum contract. The DSL layer
// keeps its own copy of the block constant (kQwenQ8BlockSize); the
// static_assert below is what keeps the two in lockstep.
#include "qwen/common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <luisa/dsl/sugar.h>

namespace luisa::qwen::dsl {

using namespace luisa::compute;

// The DSL layer keeps its own copy of the Q8_0 block geometry so dsl_ops.h stays
// free of the host contract header; these keep the two in lockstep.
static_assert(kQwenQ8BlockSize == ::luisa::qwen::kQ8_0BlockSize,
              "DSL q8_0 block size must match the GGUF Q8_0 block geometry");
static_assert(kQwenQ8BlockBytes == ::luisa::qwen::kQ8_0BlockBytes,
              "DSL q8_0 block size must match the GGUF Q8_0 block payload");

// ---------------------------------------------------------------------------
// Status helpers (shared ocr::Status/Result contract; see dsl_ops.h).
// ---------------------------------------------------------------------------
namespace {

[[nodiscard]] Status err_invalid(luisa::string msg) {
    return Status::error(StatusCode::invalid_argument, std::move(msg));
}
[[nodiscard]] Status err_overflow(luisa::string msg) {
    return Status::error(StatusCode::invalid_argument, std::move(msg), "overflow");
}

// Checked uint64 -> uint32 for kernel scalar arguments.
[[nodiscard]] Result<uint32_t> checked_u32(uint64_t v, const char *what) {
    if (v > 0xFFFFFFFFull) {
        return Result<uint32_t>::failure(err_overflow(luisa::string{what} + " exceeds the uint32 kernel envelope"));
    }
    return Result<uint32_t>::success(static_cast<uint32_t>(v));
}

// Checked product (fail-closed on uint64 overflow AND the uint32 envelope).
[[nodiscard]] Result<uint64_t> checked_product_u32(uint64_t a, uint64_t b, const char *what) {
    if (a != 0u && b > 0xFFFFFFFFull / a) {
        return Result<uint64_t>::failure(err_overflow(luisa::string{what} + " product exceeds the uint32 envelope"));
    }
    return Result<uint64_t>::success(a * b);
}

}// namespace

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------
[[nodiscard]] float QwenDslConfig::attn_scale() const noexcept {
    return 1.0f / std::sqrt(static_cast<float>(attn_head_dim));
}
[[nodiscard]] float QwenDslConfig::gdn_q_scale() const noexcept {
    return 1.0f / std::sqrt(static_cast<float>(gdn_head_dim));
}

[[nodiscard]] Status validate_qwen_config(const QwenDslConfig &cfg) noexcept {
    if (cfg.hidden == 0u || cfg.ffn == 0u || cfg.vocab == 0u) {
        return err_invalid("qwen config: hidden/ffn/vocab must be nonzero");
    }
    if (cfg.q_heads == 0u || cfg.kv_heads == 0u || cfg.q_heads % cfg.kv_heads != 0u) {
        return err_invalid("qwen config: q_heads must be a positive multiple of kv_heads (GQA)");
    }
    if (cfg.attn_head_dim == 0u || cfg.attn_head_dim % 32u != 0u || cfg.attn_head_dim > 256u) {
        return err_invalid("qwen config: attn_head_dim must be a multiple of 32 in [32,256] (decode shared-tile bound)");
    }
    if (cfg.rope_dim == 0u || cfg.rope_dim % 2u != 0u || cfg.rope_dim > cfg.attn_head_dim) {
        return err_invalid("qwen config: rope_dim must be even and <= attn_head_dim");
    }
    if (cfg.gdn_qk_heads == 0u || cfg.gdn_v_heads == 0u || cfg.gdn_v_heads % cfg.gdn_qk_heads != 0u) {
        return err_invalid("qwen config: gdn_v_heads must be a positive multiple of gdn_qk_heads");
    }
    if (cfg.gdn_head_dim == 0u || cfg.gdn_head_dim % 32u != 0u || cfg.gdn_head_dim > 128u) {
        return err_invalid("qwen config: gdn_head_dim must be a multiple of 32 in [32,128] (one state column per lane)");
    }
    if (cfg.conv_kernel < 2u) {
        return err_invalid("qwen config: conv_kernel must be >= 2");
    }
    if (!(cfg.norm_eps > 0.0f) || !(cfg.l2_eps > 0.0f) || !(cfg.rope_theta > 0.0f)) {
        return err_invalid("qwen config: eps/theta must be positive");
    }
    if (cfg.fp8_group == 0u) {
        return err_invalid("qwen config: fp8_group must be nonzero");
    }
    // Fixed per-layer shapes must fit the uint32 index envelope (checked).
    const uint64_t pairs[][2] = {
        {cfg.vocab, cfg.hidden},
        {cfg.attn_q_proj_n(), cfg.hidden},
        {cfg.attn_kv_n(), cfg.hidden},
        {cfg.hidden, cfg.attn_o_k()},
        {cfg.gdn_qkv_n(), cfg.hidden},
        {cfg.gdn_v_n(), cfg.hidden},
        {cfg.hidden, cfg.gdn_v_n()},
        {cfg.ffn, cfg.hidden},
    };
    for (auto &p : pairs) {
        if (!checked_product_u32(p[0], p[1], "qwen config weight shape").ok()) {
            return err_overflow("qwen config: a fixed per-layer weight shape exceeds uint32");
        }
    }
    const auto state_n = checked_product_u32(cfg.gdn_v_heads, cfg.gdn_head_dim, "qwen config state");
    if (!state_n.ok() || !checked_product_u32(*state_n.value, cfg.gdn_head_dim, "qwen config state").ok()) {
        return err_overflow("qwen config: GDN state shape exceeds uint32");
    }
    return Status::success();
}

[[nodiscard]] luisa::string_view qwen_weight_format_name(QwenWeightFormat format) noexcept {
    switch (format) {
        case QwenWeightFormat::bf16: return "bf16";
        case QwenWeightFormat::fp8_e4m3: return "fp8_e4m3";
        case QwenWeightFormat::q8_0: return "q8_0";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Host bit helpers (exact parity with the device decode paths)
// ---------------------------------------------------------------------------
[[nodiscard]] uint16_t host_f32_to_bf16_bits(float v) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &v, 4u);
    if (((bits >> 23u) & 0xFFu) == 0xFFu && (bits & 0x7FFFFFu) != 0u) {
        // NaN: keep a quiet NaN payload (truncate, force quiet bit).
        return static_cast<uint16_t>((bits >> 16u) | 0x0040u);
    }
    // Round-to-nearest-even on the discarded low 16 bits.
    const uint32_t rounding_bias = 0x7FFFu + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16u);
}

[[nodiscard]] float host_bf16_bits_to_f32(uint16_t bits) noexcept {
    const uint32_t out = static_cast<uint32_t>(bits) << 16u;
    float f;
    std::memcpy(&f, &out, 4u);
    return f;
}

[[nodiscard]] float host_fp8_e4m3_bits_to_f32(uint8_t code) noexcept {
    const uint32_t sign = (code >> 7u) & 1u;
    const uint32_t exp = (code >> 3u) & 0xFu;
    const uint32_t mant = code & 0x7u;
    uint32_t out;
    if (exp == 0u) {
        if (mant == 0u) {
            out = sign << 31u;// +/- zero
        } else {
            // Subnormal: value = mant * 2^-9. Normalize.
            uint32_t m = mant;
            uint32_t e = 0u;
            while ((m & 0x8u) == 0u) {
                m <<= 1u;
                e += 1u;
            }
            m &= 0x7u;
            out = (sign << 31u) | ((127u - 7u - e + 1u) << 23u) | (m << 20u);
        }
    } else if (exp == 0xFu && mant == 0x7u) {
        out = (sign << 31u) | 0x7FC00000u;// NaN (OCP E4M3FN: no infinities)
    } else {
        out = (sign << 31u) | ((exp + 127u - 7u) << 23u) | (mant << 20u);
    }
    float f;
    std::memcpy(&f, &out, 4u);
    return f;
}

[[nodiscard]] uint8_t host_f32_to_fp8_e4m3_bits(float v) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &v, 4u);
    const uint32_t sign = (bits >> 31u) & 1u;
    const int32_t exp = static_cast<int32_t>((bits >> 23u) & 0xFFu);
    const uint32_t mant = bits & 0x7FFFFFu;
    if (exp == 0xFF) {
        // NaN -> E4M3 NaN; Inf -> saturate to max finite (448).
        if (mant != 0u) { return static_cast<uint8_t>((sign << 7u) | 0x7Fu); }
        return static_cast<uint8_t>((sign << 7u) | 0x7Eu);
    }
    const int32_t unbiased = exp - 127;
    if (exp == 0) {
        // FP32 subnormal: magnitude < 2^-126, far below the E4M3 subnormal
        // floor 2^-9; rounds to zero (ties-to-even keeps zero).
        return static_cast<uint8_t>(sign << 7u);
    }
    // Value = 1.mant * 2^unbiased (or subnormal E4M3 when unbiased < -6).
    // E4M3 normal: exp field e = unbiased + 7 in [1,15], 3 mantissa bits.
    // E4M3 subnormal: e = 0, value = m/8 * 2^-6, m in [0,7].
    int32_t e;
    uint32_t m23;// mantissa in units of 2^-23 relative to the leading 1
    if (unbiased >= -6) {
        e = unbiased + 7;
        m23 = mant;
    } else {
        // Subnormal: shift the implicit-1 mantissa right by (-6 - unbiased).
        const int32_t shift = -6 - unbiased;// >= 1
        if (shift > 24) { return static_cast<uint8_t>(sign << 7u); }
        const uint32_t full = (1u << 23u) | mant;
        // Round-to-nearest-even at bit position (shift + 20).
        const int32_t drop = shift + 20;
        if (drop >= 32) { return static_cast<uint8_t>(sign << 7u); }
        const uint32_t keep = full >> drop;
        const uint32_t rem = full & ((1u << drop) - 1u);
        const uint32_t half = 1u << (drop - 1u);
        uint32_t rounded = keep;
        if (rem > half || (rem == half && (keep & 1u) != 0u)) { rounded += 1u; }
        if (rounded >= 8u) {
            // Promoted to the smallest normal.
            return static_cast<uint8_t>((sign << 7u) | (1u << 3u));
        }
        return static_cast<uint8_t>((sign << 7u) | rounded);
    }
    if (e >= 15) {
        // e == 15 with mantissa 7 is NaN; saturate finite overflow to 0x7E
        // (e=15, m=6 -> 448). Round first: values >= 464 (midpoint 448..480)
        // would round to 480 which is not representable -> saturate.
        if (e > 15) { return static_cast<uint8_t>((sign << 7u) | 0x7Eu); }
        // e == 15: round mantissa to 3 bits, cap at 6.
        const uint32_t keep = m23 >> 20u;
        const uint32_t rem = m23 & 0xFFFFFu;
        uint32_t rounded = keep;
        if (rem > 0x80000u || (rem == 0x80000u && (keep & 1u) != 0u)) { rounded += 1u; }
        if (rounded > 6u) { rounded = 6u; }
        return static_cast<uint8_t>((sign << 7u) | (15u << 3u) | rounded);
    }
    // Normal range: round mantissa to 3 bits (RNE), carry into exponent.
    const uint32_t keep = m23 >> 20u;
    const uint32_t rem = m23 & 0xFFFFFu;
    uint32_t rounded = keep;
    if (rem > 0x80000u || (rem == 0x80000u && (keep & 1u) != 0u)) { rounded += 1u; }
    if (rounded == 8u) {
        rounded = 0u;
        e += 1;
        if (e >= 15) {
            // 256 * 2^8 = ... e==15,m==0 is 256? No: e=15 -> 2^8=256, m=0 -> 256.
            // Actually e=15,m=0 = 2^(15-7)=256; fine unless it overflowed past
            // the max finite: e==15 && m>6 saturates.
            return static_cast<uint8_t>((sign << 7u) | (15u << 3u));
        }
    }
    if (e == 15 && rounded > 6u) { rounded = 6u; }
    return static_cast<uint8_t>((sign << 7u) | (static_cast<uint32_t>(e) << 3u) | rounded);
}

[[nodiscard]] Result<uint64_t> packed_matrix_words(QwenWeightFormat format, uint64_t k, uint64_t n) {
    const auto elems = checked_product_u32(k, n, "packed matrix");
    if (!elems.ok()) { return Result<uint64_t>::failure(elems.status); }
    const uint64_t e = *elems.value;
    switch (format) {
        case QwenWeightFormat::bf16:
            if ((e & 1u) != 0u) {
                return Result<uint64_t>::failure(err_invalid("bf16 packing requires an even element count (K*N)"));
            }
            return Result<uint64_t>::success(e / 2u);
        case QwenWeightFormat::fp8_e4m3:
            return Result<uint64_t>::success((e + 3u) / 4u);
        case QwenWeightFormat::q8_0:
            // Whole 32-weight blocks are one packed group of 8 words, so K must
            // be a multiple of the GGUF block size: no row may straddle a block.
            if (auto st = validate_qwen_q8_0_geometry(k, kQwenQ8BlockSize); !st.ok()) {
                return Result<uint64_t>::failure(std::move(st));
            }
            return Result<uint64_t>::success(e / 4u);
        case QwenWeightFormat::w4:
            // One word holds 8 codes, so a row of K weights is exactly K/8
            // words; K must be a multiple of 8 (whole words).
            if (k == 0u || k % kQwenW4CodesPerWord != 0u) {
                return Result<uint64_t>::failure(err_invalid(
                    "w4 packing requires K % 8 == 0 (whole packed words)"));
            }
            return Result<uint64_t>::success(e / 8u);
    }
    return Result<uint64_t>::failure(err_invalid("unknown weight format"));
}

[[nodiscard]] Result<uint64_t> fp8_scale_elements(uint64_t k, uint64_t n, uint32_t group) {
    if (group == 0u) { return Result<uint64_t>::failure(err_invalid("fp8 group must be nonzero")); }
    const uint64_t groups_per_row = (k + group - 1u) / group;
    return checked_product_u32(groups_per_row, n, "fp8 scales");
}

[[nodiscard]] Result<uint64_t> q8_0_scale_elements(uint64_t k, uint64_t n) {
    if (auto st = validate_qwen_q8_0_geometry(k, kQwenQ8BlockSize); !st.ok()) {
        return Result<uint64_t>::failure(std::move(st));
    }
    return checked_product_u32(k / kQwenQ8BlockSize, n, "q8_0 scales");
}

[[nodiscard]] Result<uint64_t> weight_scale_elements(QwenWeightFormat format, uint64_t k,
                                                     uint64_t n, uint32_t group) {
    if (format == QwenWeightFormat::bf16) { return Result<uint64_t>::success(0u); }
if (format == QwenWeightFormat::q8_0) { return q8_0_scale_elements(k, n); }
if (format == QwenWeightFormat::w4) {
    if (auto st = validate_qwen_w4_geometry(k, group); !st.ok()) {
        return Result<uint64_t>::failure(std::move(st));
    }
    return checked_product_u32(k / group, n, "w4 scales");
}
return fp8_scale_elements(k, n, group);
}

[[nodiscard]] Status validate_qwen_q8_0_geometry(uint64_t k, uint32_t group) noexcept {
    if (group != kQwenQ8BlockSize) {
        return err_invalid("q8_0 requires the GGUF block group of 32 along K");
    }
    if (k == 0u || k % kQwenQ8BlockSize != 0u) {
        return err_invalid(
            "q8_0 requires K % 32 == 0 (whole 32-weight blocks and whole 8-word groups)");
    }
    return Status::success();
}

[[nodiscard]] Status validate_qwen_w4_geometry(uint64_t k, uint32_t group) noexcept {
    if (group != 32u && group != 64u && group != 128u) {
        return err_invalid("w4 requires a scale group in {32, 64, 128}");
    }
    if (k == 0u) {
        return err_invalid("w4 requires K > 0");
    }
    if (k % kQwenW4CodesPerWord != 0u) {
        return err_invalid("w4 requires K % 8 == 0 (whole packed words)");
    }
    if (k % group != 0u) {
        return err_invalid("w4 requires K % group == 0 (whole scale groups)");
    }
    return Status::success();
}

[[nodiscard]] float host_w4_code_to_f32(uint32_t nibble) noexcept {
    // Exact two's-complement sign extension of a 4-bit code: ((c ^ 8) - 8).
    return static_cast<float>(static_cast<int>((nibble & 0xFu) ^ 0x8u) - 8);
}

namespace {
// Deterministic nearest-int4 code for one value against a float scale:
// round half away from zero on the double quotient, clamp to [-8, 7], return
// the 4-bit two's-complement nibble. Identical to the W4 packer in common.cpp
// (the offline packer and the host reference must agree bit for bit).
[[nodiscard]] uint32_t host_w4_quantize_code(double value, double scale) noexcept {
    if (!(scale > 0.0)) { return 0u; }
    const double q = value / scale;
    const double r = q >= 0.0 ? std::floor(q + 0.5) : std::ceil(q - 0.5);
    const double clamped = r > 7.0 ? 7.0 : (r < -8.0 ? -8.0 : r);
    return static_cast<uint32_t>(static_cast<int>(clamped)) & 0xFu;
}
}// namespace

[[nodiscard]] Status host_pack_w4_matrix(luisa::span<const float> w,
                                         uint64_t k, uint64_t n, uint32_t group,
                                         float min_ratio, float max_ratio, uint32_t steps,
                                         luisa::vector<uint32_t> &words,
                                         luisa::vector<float> &scales,
                                         float *mean_ratio,
                                         double *mse,
                                         double *rtn_mse) {
    const auto wc = packed_matrix_words(QwenWeightFormat::w4, k, n);
    if (!wc.ok()) { return wc.status; }
    if (w.size() != k * n) { return err_invalid("host_pack_w4_matrix: weight span size mismatch"); }
    if (auto st = validate_qwen_w4_geometry(k, group); !st.ok()) { return st; }
    if (!(min_ratio > 0.0f) || !(max_ratio <= 1.0f) || min_ratio > max_ratio || steps < 2u) {
        return err_invalid("host_pack_w4_matrix: invalid clipping search config");
    }
    const auto sc = weight_scale_elements(QwenWeightFormat::w4, k, n, group);
    if (!sc.ok()) { return sc.status; }
    for (uint64_t i = 0; i < k * n; ++i) {
        if (!std::isfinite(w[static_cast<size_t>(i)])) {
            return err_invalid("host_pack_w4_matrix: input is not finite");
        }
    }
    words.assign(static_cast<size_t>(*wc.value), 0u);
    scales.assign(static_cast<size_t>(*sc.value), 0.0f);
    const uint64_t gpr = k / group;
    const double lo = static_cast<double>(min_ratio);
    const double hi = static_cast<double>(max_ratio);
    const auto ratio_at = [&](uint32_t i) {
        return lo + static_cast<double>(i) * (hi - lo) / static_cast<double>(steps - 1u);
    };
    double ratio_sum = 0.0;
    double sq_error = 0.0;
    double rtn_sq_error = 0.0;
    luisa::vector<double> group_values(static_cast<size_t>(group));
    luisa::vector<uint32_t> best_codes(static_cast<size_t>(group));
    for (uint64_t row = 0; row < n; ++row) {
        for (uint64_t g = 0; g < gpr; ++g) {
            const uint64_t begin = row * k + g * group;
            double max_abs = 0.0;
            for (uint32_t i = 0; i < group; ++i) {
                const double v = static_cast<double>(w[static_cast<size_t>(begin + i)]);
                group_values[i] = v;
                const double a = std::fabs(v);
                if (a > max_abs) { max_abs = a; }
            }
            const uint64_t scale_index = row * gpr + g;
            const double base = max_abs / 7.0;
            double best_mse = std::numeric_limits<double>::infinity();
            float best_scale = 0.0f;
            double best_ratio = 1.0;
            if (base > 0.0) {
                for (uint32_t si = 0; si < steps; ++si) {
                    const double ratio = ratio_at(si);
                    const float scale_f = static_cast<float>(base * ratio);
                    if (!(scale_f > 0.0f)) { continue; }
                    const double scale_d = static_cast<double>(scale_f);
                    double err = 0.0;
                    for (uint32_t i = 0; i < group; ++i) {
                        const float dq_code = host_w4_code_to_f32(
                            host_w4_quantize_code(group_values[i], scale_d));
                        const double dq = static_cast<double>(dq_code) * scale_d;
                        const double e = group_values[i] - dq;
                        err += e * e;
                    }
                    if (err < best_mse) {// strict: ties keep the smaller ratio
                        best_mse = err;
                        best_scale = scale_f;
                        best_ratio = ratio;
                        for (uint32_t i = 0; i < group; ++i) {
                            best_codes[i] = host_w4_quantize_code(group_values[i], scale_d);
                        }
                    }
                }
            }
            if (base > 0.0 && !(best_scale > 0.0f)) {
                return err_invalid("host_pack_w4_matrix: clipping search produced no valid scale");
            }
            scales[static_cast<size_t>(scale_index)] = best_scale;
            ratio_sum += best_ratio;
            if (base > 0.0) { sq_error += best_mse; }
            // Plain max-abs RTN reference (same denominator, no clipping).
            for (uint32_t i = 0; i < group; ++i) {
                const float dq_code = host_w4_code_to_f32(
                    host_w4_quantize_code(group_values[i], base));
                const double dq = static_cast<double>(dq_code) * base;
                const double e = group_values[i] - dq;
                rtn_sq_error += e * e;
            }
            for (uint32_t i = 0; i < group; i += kQwenW4CodesPerWord) {
                uint32_t word = 0u;
                for (uint32_t j = 0; j < kQwenW4CodesPerWord; ++j) {
                    word |= (best_codes[i + j] & 0xFu) << (4u * j);
                }
                words[static_cast<size_t>((begin + i) / kQwenW4CodesPerWord)] = word;
            }
        }
    }
    if (mean_ratio != nullptr) {
        *mean_ratio = static_cast<float>(ratio_sum / static_cast<double>(k * n / group));
    }
    if (mse != nullptr) { *mse = sq_error / static_cast<double>(k * n); }
    if (rtn_mse != nullptr) { *rtn_mse = rtn_sq_error / static_cast<double>(k * n); }
    return Status::success();
}

[[nodiscard]] Status host_pack_matrix(QwenWeightFormat format, luisa::span<const float> w,
                                         uint64_t k, uint64_t n, uint32_t group,
                                         luisa::vector<uint32_t> &words,
                                         luisa::vector<float> &scales) {
    const auto wc = packed_matrix_words(format, k, n);
    if (!wc.ok()) { return wc.status; }
    if (w.size() != k * n) { return err_invalid("host_pack_matrix: weight span size mismatch"); }
    if (format == QwenWeightFormat::q8_0) {
        // Refused on purpose: a Q8_0 matrix must come from the checkpoint's own
        // int8 block codes (host_pack_q8_0_codes / pack_q8_0_blocks). Quantizing
        // FP32 here would be exactly the silent double quantization the loader
        // contract forbids.
        return err_invalid(
            "host_pack_matrix does not quantize to q8_0: upload the GGUF int8 block "
            "payload (pack_q8_0_blocks) or use host_pack_q8_0_codes");
    }
    if (format == QwenWeightFormat::w4) {
        // Canonical clipping search; host_pack_w4_matrix exposes the full config.
        return host_pack_w4_matrix(w, k, n, group, 0.85f, 1.0f, 31u, words, scales);
    }
    words.assign(static_cast<size_t>(*wc.value), 0u);
    scales.clear();
    if (format == QwenWeightFormat::bf16) {
        for (uint64_t i = 0; i < k * n; i += 2u) {
            const uint32_t lo = host_f32_to_bf16_bits(w[i]);
            const uint32_t hi = host_f32_to_bf16_bits(w[i + 1u]);
            words[static_cast<size_t>(i / 2u)] = lo | (hi << 16u);
        }
        return Status::success();
    }
    if (group == 0u) { return err_invalid("host_pack_matrix: fp8 group must be nonzero"); }
    const auto sc = fp8_scale_elements(k, n, group);
    if (!sc.ok()) { return sc.status; }
    scales.resize(static_cast<size_t>(*sc.value));
    const uint64_t gpr = (k + group - 1u) / group;
    for (uint64_t row = 0; row < n; row++) {
        for (uint64_t g = 0; g < gpr; g++) {
            const uint64_t begin = g * group;
            const uint64_t end = begin + group < k ? begin + group : k;
            double max_abs = 0.0;
            for (uint64_t i = begin; i < end; i++) {
                const double a = std::fabs(static_cast<double>(w[row * k + i]));
                if (a > max_abs) { max_abs = a; }
            }
            // scale = max_abs/448; zero groups get scale 0 (codes all zero).
            const float scale = static_cast<float>(max_abs / 448.0);
            scales[static_cast<size_t>(row * gpr + g)] = scale;
            const double inv = scale > 0.0f ? 1.0 / static_cast<double>(scale) : 0.0;
            for (uint64_t i = begin; i < end; i++) {
                const double q = static_cast<double>(w[row * k + i]) * inv;
                // Saturate to the finite E4M3 range before RNE conversion.
                const double clamped = q > 448.0 ? 448.0 : (q < -448.0 ? -448.0 : q);
                const uint8_t code = host_f32_to_fp8_e4m3_bits(static_cast<float>(clamped));
                const uint64_t idx = row * k + i;
                words[static_cast<size_t>(idx / 4u)] |= static_cast<uint32_t>(code) << ((idx & 3u) * 8u);
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status host_pack_q8_0_codes(luisa::span<const uint8_t> codes,
                                          luisa::span<const uint16_t> scale_bits,
                                          uint64_t k, uint64_t n,
                                          luisa::vector<uint32_t> &words,
                                          luisa::vector<float> &scales) {
    const auto wc = packed_matrix_words(QwenWeightFormat::q8_0, k, n);
    if (!wc.ok()) { return wc.status; }
    const auto sc = q8_0_scale_elements(k, n);
    if (!sc.ok()) { return sc.status; }
    if (codes.size() != k * n) { return err_invalid("host_pack_q8_0_codes: code span size mismatch"); }
    if (scale_bits.size() != *sc.value) {
        return err_invalid("host_pack_q8_0_codes: scale span size mismatch");
    }
    // Build the raw 34-byte block payload the loader reads from the GGUF and
    // reuse its exact split, so tests and the device upload cannot diverge.
    const uint64_t blocks = k * n / kQwenQ8BlockSize;
    luisa::vector<std::byte> payload(static_cast<size_t>(blocks * kQwenQ8BlockBytes));
    for (uint64_t b = 0; b < blocks; ++b) {
        const uint16_t bits = scale_bits[static_cast<size_t>(b)];
        payload[static_cast<size_t>(b * kQwenQ8BlockBytes)] =
            static_cast<std::byte>(bits & 0xffu);
        payload[static_cast<size_t>(b * kQwenQ8BlockBytes + 1u)] =
            static_cast<std::byte>(bits >> 8u);
        for (uint64_t i = 0; i < kQwenQ8BlockSize; ++i) {
            payload[static_cast<size_t>(b * kQwenQ8BlockBytes + 2u + i)] =
                static_cast<std::byte>(codes[static_cast<size_t>(b * kQwenQ8BlockSize + i)]);
        }
    }
    words.assign(static_cast<size_t>(*wc.value), 0u);
    scales.assign(static_cast<size_t>(*sc.value), 0.0f);
    return pack_q8_0_blocks(payload, k * n,
                            luisa::span<uint32_t>{words.data(), words.size()},
                            luisa::span<float>{scales.data(), scales.size()});
}

[[nodiscard]] float host_decode_packed(QwenWeightFormat format, luisa::span<const uint32_t> words,
                                       luisa::span<const float> scales, uint64_t k,
                                       uint32_t group, uint64_t index) noexcept {
    if (format == QwenWeightFormat::bf16) {
        const uint32_t word = words[static_cast<size_t>(index >> 1u)];
        const uint32_t bits = (word >> ((index & 1u) * 16u)) & 0xFFFFu;
        return host_bf16_bits_to_f32(static_cast<uint16_t>(bits));
    }
    const uint64_t row = index / k;
    const uint64_t gpr = (group == 0u) ? 1u : (k + group - 1u) / group;
    if (format == QwenWeightFormat::w4) {
        const uint32_t word = words[static_cast<size_t>(index >> 3u)];
        const uint32_t code = (word >> ((index & 7u) * 4u)) & 0xFu;
        const float scale = scales[static_cast<size_t>(row * gpr + (index % k) / group)];
        return host_w4_code_to_f32(code) * scale;
    }
    const uint32_t word = words[static_cast<size_t>(index >> 2u)];
    const uint32_t code = (word >> ((index & 3u) * 8u)) & 0xFFu;
    const float scale = scales[static_cast<size_t>(row * gpr + (index % k) / group)];
    if (format == QwenWeightFormat::q8_0) {
        // The GGUF Q8_0 code IS a signed int8: exact sign extension * block scale.
        return int8_to_f32(static_cast<uint8_t>(code)) * scale;
    }
    return host_fp8_e4m3_bits_to_f32(static_cast<uint8_t>(code)) * scale;
}

// ---------------------------------------------------------------------------
// Launch planning
// ---------------------------------------------------------------------------
[[nodiscard]] Result<QwenLaunch1D> plan_qwen_threads(uint64_t count, uint32_t block) {
    if (block == 0u) { return Result<QwenLaunch1D>::failure(err_invalid("block size must be nonzero")); }
    QwenLaunch1D plan;
    plan.block = block;
    plan.work_items = count;
    if (count == 0u) { return Result<QwenLaunch1D>::success(plan); }
    if (count > 0xFFFFFFFFull) {
        return Result<QwenLaunch1D>::failure(err_overflow("work item count exceeds uint32"));
    }
    const uint64_t groups = (count + block - 1u) / block;
    if (groups <= kQwenMaxGroupsPerDim) {
        plan.total_x = static_cast<uint32_t>(groups * block);
        plan.total_y = 1u;
    } else {
        // Flatten: x stays one block-row wide times a bounded group count,
        // y carries the remaining block rows. Keep total_x a multiple of
        // block and both group counts inside the portable envelope.
        const uint64_t gx = kQwenMaxGroupsPerDim;
        const uint64_t gy = (groups + gx - 1u) / gx;
        if (gy > kQwenMaxGroupsPerDim) {
            return Result<QwenLaunch1D>::failure(err_overflow("flattened grid exceeds the 2D group envelope"));
        }
        const uint64_t padded_threads = gx * block * gy;
        if (padded_threads > 0xFFFFFFFFull) {
            return Result<QwenLaunch1D>::failure(err_overflow("flattened total-thread count exceeds uint32"));
        }
        plan.total_x = static_cast<uint32_t>(gx * block);
        plan.total_y = static_cast<uint32_t>(gy);
    }
    return Result<QwenLaunch1D>::success(plan);
}

[[nodiscard]] Result<QwenLaunch2D> plan_qwen_threads_2d(uint64_t rows, uint64_t cols,
                                                          uint32_t block_x, uint32_t block_y) {
    if (block_x == 0u || block_y == 0u) {
        return Result<QwenLaunch2D>::failure(err_invalid("block sizes must be nonzero"));
    }
    QwenLaunch2D plan;
    plan.rows = rows;
    plan.cols = cols;
    plan.block_x = block_x;
    plan.block_y = block_y;
    if (rows == 0u || cols == 0u) { return Result<QwenLaunch2D>::success(plan); }
    const auto c32 = checked_u32(cols, "cols");
    if (!c32.ok()) { return Result<QwenLaunch2D>::failure(c32.status); }
    const uint64_t gx = (cols + block_x - 1u) / block_x;
    const uint64_t gy = (rows + block_y - 1u) / block_y;
    if (gx > kQwenMaxGroupsPerDim) {
        return Result<QwenLaunch2D>::failure(err_overflow("column group count exceeds the portable envelope"));
    }
    plan.padded_cols = static_cast<uint32_t>(gx * block_x);
    if (gy <= kQwenMaxGroupsPerDim) {
        plan.total_x = static_cast<uint32_t>(gx * block_x);
        plan.total_y = static_cast<uint32_t>(gy * block_y);
        plan.folded = false;
    } else {
        // Fold block-rows into x: the linear id spans padded_cols columns
        // per row; the kernel recovers row = linear / padded_cols.
        const uint64_t total_threads = static_cast<uint64_t>(plan.padded_cols) * gy * block_y;
        const uint64_t groups = (total_threads + block_x - 1u) / block_x;
        if (groups > static_cast<uint64_t>(kQwenMaxGroupsPerDim) * kQwenMaxGroupsPerDim) {
            return Result<QwenLaunch2D>::failure(err_overflow("folded grid exceeds the 2D group envelope"));
        }
        const uint64_t fx = kQwenMaxGroupsPerDim;
        const uint64_t fy = (groups + fx - 1u) / fx;
        plan.total_x = static_cast<uint32_t>(fx * block_x);
        plan.total_y = static_cast<uint32_t>(fy * block_y);
        plan.folded = true;
    }
    return Result<QwenLaunch2D>::success(plan);
}

// Warp-per-row plan: `rows` warps of 32 lanes, blocks of `block` threads
// (multiple of 32). work_items = rows (the kernel guards warp_id < rows).
namespace {
[[nodiscard]] Result<QwenLaunch1D> plan_warp_rows(uint64_t rows, uint32_t block) {
    if (block == 0u || block % 32u != 0u) {
        return Result<QwenLaunch1D>::failure(err_invalid("warp-row block must be a nonzero multiple of 32"));
    }
    const auto threads = checked_product_u32(rows, 32u, "warp rows");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto plan = plan_qwen_threads(*threads.value, block);
    if (!plan.ok()) { return Result<QwenLaunch1D>::failure(plan.status); }
    plan.value->work_items = rows;
    plan.value->warp_size = 32u;
    return plan;
}
}// namespace

// ---------------------------------------------------------------------------
// Device-side shared helpers (recorded into the AST; no host state).
// ---------------------------------------------------------------------------
namespace {

// Linear id for a (possibly flattened) 1D launch. dispatch_size() is the
// TOTAL thread count per dimension.
[[nodiscard]] auto qwen_linear_id() noexcept {
    return dispatch_y() * dispatch_size().x + dispatch_x();
}

// (row, col) recovery for a folded 2D launch (padded_cols stride).
[[nodiscard]] auto qwen_folded_row() noexcept { return qwen_linear_id(); }

// Exact bf16 -> fp32.
[[nodiscard]] auto qwen_bf16_to_f32(Var<uint32_t> bits) noexcept {
    return as<float>(bits << 16u);
}

// Exact E4M3 -> fp32 (finite codes incl. subnormals; S.1111.111 -> NaN).
[[nodiscard]] auto qwen_fp8_e4m3_to_f32(Var<uint32_t> code) noexcept {
    Var<uint32_t> sign = (code >> 7u) & 1u;
    Var<uint32_t> exp = (code >> 3u) & 0xFu;
    Var<uint32_t> mant = code & 0x7u;
    Var<uint32_t> out_bits = def(0u);
    $if (exp == 0u) {
        $if (mant != 0u) {
            Var<uint32_t> m = def(mant);
            Var<uint32_t> e = def(0u);
            $while ((m & 0x8u) == 0u) {
                m = m << 1u;
                e += 1u;
            };
            m = m & 0x7u;
            out_bits = (sign << 31u) | ((127u - 7u - e + 1u) << 23u) | (m << 20u);
        } $else {
            out_bits = sign << 31u;
        };
    } $elif (exp == 0xFu & mant == 0x7u) {
        out_bits = (sign << 31u) | 0x7FC00000u;
    } $else {
        out_bits = (sign << 31u) | ((exp + 127u - 7u) << 23u) | (mant << 20u);
    };
    return as<float>(out_bits);
}

// BRANCHLESS, LUT-FREE E4M3 -> fp32 (opt_plan/09 N2). Bit-identical to
// qwen_fp8_e4m3_to_f32 above for all 256 codes; it exists so the FP8 decode
// kernels do not need a 256-entry Shared<float> LUT (and its sync_block, and its
// random-access bank conflicts).
//
// Construction, per code field (sign s, exponent e in [0,15], mantissa m in [0,7]):
//  * normal (e != 0): the reference writes (s<<31)|((e+120)<<23)|(m<<20). E4M3
//    has bias 7 and no infinity, so e == 15 with m != 7 is simply a large finite
//    value -- the same formula covers it.
//  * subnormal (e == 0, m != 0): the reference normalizes m and emits
//    (s<<31)|((118-e')<<23)|(m'<<20), i.e. the value m*2^-9. The classic
//    "magic constant" identity gives that directly and exactly:
//    as<float>((s<<31)|0x3C800000|(m<<20)) == s*(2^-6 + m*2^-9) and the
//    subtrahend as<float>((s<<31)|0x3C800000) == s*2^-6, so their difference is
//    s*m*2^-9. Both operands are exactly representable and the subtraction is
//    exact (m*2^-9 needs 3 mantissa bits below 2^-6).
//  * signed zero (e == 0, m == 0): the difference above is +0, so the sign bit
//    is OR-ed back in (harmless for every other code, which already carries it).
//  * NaN (e == 15, m == 7): S.1111.111 maps to (s<<31)|0x7FC00000, exactly the
//    reference's NaN, selected with one predicated `ite` (no branch).
[[nodiscard]] auto qwen_fp8_e4m3_to_f32_branchless(Var<uint32_t> code) noexcept {
    constexpr auto kMagic = 0x3C800000u;// 2^-6
    const auto sign = code >> 7u;
    const auto sign_bits = sign << 31u;
    const auto exp = (code >> 3u) & 0xFu;
    const auto mant = code & 0x7u;
    // Subnormal magnitude: (2^-6 + m*2^-9) - 2^-6 == m*2^-9, exact in FP32.
    const auto mag = as<float>(kMagic | (mant << 20u)) - as<float>(kMagic);
    // ... with the code's sign (negating +0 gives the exact -0 of code 0x80).
    const auto sub = ite(sign != 0u, -mag, mag);
    // Normal path (also the e == 15, m != 7 finite values: E4M3 FN has no inf).
    const auto normal = as<float>(sign_bits | ((exp + 120u) << 23u) | (mant << 20u));
    const auto finite = ite(exp == 0u, sub, normal);
    // S.1111.111 -> NaN, exactly like the reference conversion.
    return ite(exp == 0xFu & mant == 0x7u, as<float>(sign_bits | 0x7FC00000u), finite);
}

// SIGNED int8 code -> fp32 (Q8_0). The GGUF Q8_0 code IS a two's-complement
// int8, so the value is exactly ((code ^ 0x80) - 128): two integer ops, no
// table, no rounding, and identical on every backend (no reliance on an
// arithmetic right shift of a signed value).
[[nodiscard]] auto qwen_q8_0_to_f32(Var<uint32_t> code) noexcept {
    return cast<float>(cast<int>((code & 0xFFu) ^ 0x80u) - 128);
}

// SIGNED int4 nibble -> fp32 (W4A16). The code is a two's-complement value in
// [-8, 7], so the exact sign extension is ((code ^ 8) - 8): two integer ops, no
// table, no rounding, identical on every backend (no reliance on an arithmetic
// right shift of a signed value). Bit-identical to host_w4_code_to_f32.
[[nodiscard]] auto qwen_w4_to_f32(Var<uint32_t> code) noexcept {
    return cast<float>(cast<int>((code & 0xFu) ^ 0x8u) - 8);
}

// Kernel-name suffix per weight format. The BF16/FP8 suffixes are unchanged
// (annotations, benchmark categories and stored shader caches key on them);
// Q8_0 gets its own suffix so a profiling log can never confuse the two 8-bit
// formats.
[[nodiscard]] constexpr const char *qwen_format_name_suffix(QwenWeightFormat format) noexcept {
    return format == QwenWeightFormat::bf16 ? "_bf16"
           : format == QwenWeightFormat::fp8_e4m3 ? "_fp8"
           : format == QwenWeightFormat::w4 ? "_w4"
                                            : "_q8_0";
}

// One byte of a packed 4-codes-per-word weight -> fp32 CODE VALUE (no scale).
// fp8_e4m3: the branchless LUT-free E4M3 decode; q8_0: exact signed int8.
[[nodiscard]] auto qwen_scaled_byte_to_f32(Var<uint32_t> byte, QwenWeightFormat format) noexcept {
    if (format == QwenWeightFormat::q8_0) { return qwen_q8_0_to_f32(byte); }
    return qwen_fp8_e4m3_to_f32_branchless(byte);
}

// Decode one packed weight element. `format` is a compile-time host value.
// fp8/q8_0 scale index = row * groups_per_row + (k_index / group); the caller
// passes the row base scale offset and the within-row k index.
[[nodiscard]] auto qwen_read_weight(const BufferVar<uint32_t> &packed,
                                    const BufferVar<float> &scales,
                                    Var<uint32_t> index, Var<uint32_t> row_scale_base,
                                    Var<uint32_t> k_index, Var<uint32_t> group,
                                    QwenWeightFormat format) noexcept {
    if (format == QwenWeightFormat::bf16) {
        const auto word = packed.read(index >> 1u);
        const auto bits = (word >> ((index & 1u) * 16u)) & 0xFFFFu;
        return qwen_bf16_to_f32(bits);
    }
    if (format == QwenWeightFormat::w4) {
        // W4: 8 codes per word, low nibble first, one scale per group.
        const auto word = packed.read(index >> 3u);
        const auto code = (word >> ((index & 7u) * 4u)) & 0xFu;
        return qwen_w4_to_f32(code) * scales.read(row_scale_base + k_index / group);
    }
    const auto word = packed.read(index >> 2u);
    const auto code = (word >> ((index & 3u) * 8u)) & 0xFFu;
    const auto scale = scales.read(row_scale_base + k_index / group);
    if (format == QwenWeightFormat::q8_0) {
        return qwen_q8_0_to_f32(code) * scale;
    }
    return qwen_fp8_e4m3_to_f32(code) * scale;
}

// 32-lane butterfly sum over the warp. warp_read_lane lowers to ONE shuffle on
// every backend (__shfl_sync / WaveReadLaneAt / OpGroupNonUniformShuffle), so
// this costs 5 shuffles + 5 adds, and -- unlike the generic warp_active_sum --
// it does not consult the ACTIVE MASK. That matters on HLSL/DX, where a wave
// intrinsic issued after a divergent region (e.g. the `if (lane == 0)` write of
// a previous loop iteration) may execute with the other lanes still masked off
// and silently return a single lane's partial sum. Callers must still keep every
// lane of the warp active, which their block/warp-uniform guards guarantee.
[[nodiscard]] auto qwen_warp_butterfly_sum(Var<float> x) noexcept {
    const auto lane = warp_lane_id();
    Var<float> s = def(x);
    s += warp_read_lane(s, lane ^ 16u);
    s += warp_read_lane(s, lane ^ 8u);
    s += warp_read_lane(s, lane ^ 4u);
    s += warp_read_lane(s, lane ^ 2u);
    s += warp_read_lane(s, lane ^ 1u);
    return s;
}

[[nodiscard]] auto qwen_silu(Var<float> x) noexcept {
    return x / (1.0f + exp(-x));
}
[[nodiscard]] auto qwen_sigmoid(Var<float> x) noexcept {
    return 1.0f / (1.0f + exp(-x));
}

}// namespace

// ---------------------------------------------------------------------------
// 1. Embedding gather
// ---------------------------------------------------------------------------
[[nodiscard]] QwenEmbeddingKernel make_qwen_embedding_kernel(QwenWeightFormat format) noexcept {
    QwenEmbeddingKernel kernel = [format](BufferUInt ids, BufferUInt table, BufferFloat scales,
                                          BufferFloat out, Var<uint> tokens, Var<uint> hidden,
                                          Var<uint> vocab, Var<uint> group, Var<uint> padded_cols) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name(luisa::string{"qwen_embedding"} + qwen_format_name_suffix(format));
        // Folded launches recover (row,col) from the linear id; unfolded
        // launches use dispatch ids directly. padded_cols == dispatch_size().x
        // in the unfolded case, so one code path covers both.
        const auto linear = qwen_linear_id();
        const auto row = linear / padded_cols;
        const auto col = linear % padded_cols;
        $if (row < tokens & col < hidden) {
            const auto id = ids.read(row);
            // Defensive: an out-of-range id leaves the row untouched rather
            // than reading OOB (host validation should have rejected it).
            $if (id < vocab) {
                const auto index = id * hidden + col;
                Var<float> v = def(0.0f);
                if (format == QwenWeightFormat::bf16) {
                    const auto word = table.read(index >> 1u);
                    v = qwen_bf16_to_f32((word >> ((index & 1u) * 16u)) & 0xFFFFu);
                } else if (format == QwenWeightFormat::w4) {
                    // W4 gather row: signed int4 nibble * per-group scale.
                    const auto word = table.read(index >> 3u);
                    const auto code = (word >> ((index & 7u) * 4u)) & 0xFu;
                    const auto gpr = (hidden + group - 1u) / group;
                    v = qwen_w4_to_f32(code) * scales.read(id * gpr + col / group);
                } else {
                    const auto word = table.read(index >> 2u);
                    const auto code = (word >> ((index & 3u) * 8u)) & 0xFFu;
                    const auto gpr = (hidden + group - 1u) / group;
                    if (format == QwenWeightFormat::q8_0) {
                        // Q8_0 embedding row: signed int8 code * per-32 block scale.
                        v = qwen_q8_0_to_f32(code) * scales.read(id * gpr + col / group);
                    } else {
                        v = qwen_fp8_e4m3_to_f32(code) * scales.read(id * gpr + col / group);
                    }
                }
                out.write(row * hidden + col, v);
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch2D> plan_qwen_embedding_dispatch(uint64_t tokens, uint64_t hidden) {
    const auto n = checked_product_u32(tokens, hidden, "embedding output");
    if (!n.ok()) { return Result<QwenLaunch2D>::failure(n.status); }
    return plan_qwen_threads_2d(tokens, hidden);
}

[[nodiscard]] Status validate_qwen_embedding_inputs(luisa::span<const uint32_t> ids,
                                                       uint64_t hidden, uint64_t vocab) {
    if (hidden == 0u || vocab == 0u) { return err_invalid("embedding hidden/vocab must be nonzero"); }
    if (!checked_product_u32(hidden, vocab, "embedding table").ok()) {
        return err_overflow("embedding table exceeds uint32");
    }
    for (auto id : ids) {
        if (static_cast<uint64_t>(id) >= vocab) {
            return err_invalid("embedding token id out of range");
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// 2. Zero-centered RMSNorm (warp per row)
// ---------------------------------------------------------------------------
[[nodiscard]] QwenRmsNormKernel make_qwen_rmsnorm_kernel(QwenNormEpilogue epilogue) noexcept {
    QwenRmsNormKernel kernel = [epilogue](BufferFloat x, BufferFloat w, BufferFloat residual,
                                          BufferFloat y, BufferFloat sum_out,
                                          Var<uint> rows, Var<uint> dim, Var<float> eps) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(epilogue == QwenNormEpilogue::residual ? "qwen_rmsnorm_residual"
                 : epilogue == QwenNormEpilogue::snapshot ? "qwen_rmsnorm_snapshot"
                                                          : "qwen_rmsnorm");
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        // Block-uniform guard: every lane of a participating warp enters.
        $if (warp < rows & dim > 0u) {
            const auto base = warp * dim;
            Var<float> sumsq = def(0.0f);
            $for (i, lane, dim, 32u) {
                Var<float> v = x.read(base + i);
                if (epilogue == QwenNormEpilogue::residual) {
                    v += residual.read(base + i);
                    sum_out.write(base + i, v);
                } else if (epilogue == QwenNormEpilogue::snapshot) {
                    // Residual snapshot fused into the sum-of-squares pass:
                    // store the ORIGINAL x bit-exactly (no arithmetic on v),
                    // so the second pass below can normalize from sum_out
                    // with identical FP32 math to plain. x/w/y/sum_out must
                    // be distinct buffers (see header aliasing contract).
                    sum_out.write(base + i, v);
                }
                sumsq += v * v;
            };
            sumsq = warp_active_sum(sumsq);
            const auto inv = rsqrt(sumsq / cast<float>(dim) + eps);
            $for (i, lane, dim, 32u) {
                Var<float> v = def(0.0f);
                if (epilogue == QwenNormEpilogue::plain) {
                    v = x.read(base + i);
                } else {
                    // residual and snapshot both normalize the value they
                    // just wrote to sum_out (same-lane store->load; the
                    // warp_active_sum above separates the two passes).
                    v = sum_out.read(base + i);
                }
                // GGUF norm weights are PRE-ADDED (the llama.cpp converter
                // folds the HF zero-centered 1+w offline); multiply DIRECTLY.
                y.write(base + i, v * inv * w.read(i));
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_rmsnorm_dispatch(uint64_t rows, uint64_t dim) {
    if (dim > 0xFFFFFFFFull) { return Result<QwenLaunch1D>::failure(err_overflow("norm dim exceeds uint32")); }
    const auto n = checked_product_u32(rows, dim, "norm elements");
    if (!n.ok()) { return Result<QwenLaunch1D>::failure(n.status); }
    return plan_warp_rows(rows, 128u);
}

[[nodiscard]] QwenRmsNormKernel make_qwen35_rmsnorm_kernel(
    QwenNormEpilogue epilogue, uint32_t block_size) noexcept {
    QwenRmsNormKernel kernel = [epilogue, block_size](BufferFloat x, BufferFloat w, BufferFloat residual,
                                          BufferFloat y, BufferFloat sum_out,
                                          Var<uint> rows, Var<uint> dim, Var<float> eps) noexcept {
        set_block_size(block_size, 1u, 1u);
        set_warp_size(32u);
        set_name(epilogue == QwenNormEpilogue::residual ? "qwen35_rmsnorm_residual"
                 : epilogue == QwenNormEpilogue::snapshot ? "qwen35_rmsnorm_snapshot"
                                                          : "qwen35_rmsnorm");
        Shared<float> partial{8u};
        const auto row = qwen_linear_id() / block_size;
        const auto tid = thread_x();
        const auto lane = warp_lane_id();
        const auto wid = tid / 32u;
        $if (row < rows & dim > 0u) {
            const auto base = row * dim;
            Var<float> sumsq = def(0.0f);
            $for (i, tid, dim, block_size) {
                Var<float> v = x.read(base + i);
                if (epilogue == QwenNormEpilogue::residual) {
                    v += residual.read(base + i);
                    sum_out.write(base + i, v);
                } else if (epilogue == QwenNormEpilogue::snapshot) {
                    sum_out.write(base + i, v);
                }
                sumsq += v * v;
            };
            sumsq = warp_active_sum(sumsq);
            $if (lane == 0u) { partial[wid] = sumsq; };
            sync_block();
            // Every thread uses the same four-partial order, then normalizes
            // its own strided elements. No second cross-warp round trip.
            sumsq = 0.0f;
            for (uint32_t w = 0u; w < block_size / 32u; w++) {
                sumsq += partial.read(w);
            }
            const auto inv = rsqrt(sumsq / cast<float>(dim) + eps);
            $for (i, tid, dim, block_size) {
                const auto v = epilogue == QwenNormEpilogue::plain
                                   ? x.read(base + i)
                                   : sum_out.read(base + i);
                y.write(base + i, v * inv * w.read(i));
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_rmsnorm_dispatch(
    uint64_t rows, uint64_t dim, uint32_t block_size) {
    if (block_size != 128u && block_size != 256u) {
        return Result<QwenLaunch1D>::failure(err_invalid("qwen35 norm block must be 128 or 256"));
    }
    if (dim > 0xFFFFFFFFull) {
        return Result<QwenLaunch1D>::failure(err_overflow("qwen35 norm dim exceeds uint32"));
    }
    const auto n = checked_product_u32(rows, dim, "qwen35 norm elements");
    if (!n.ok()) { return Result<QwenLaunch1D>::failure(n.status); }
    const auto threads = checked_product_u32(rows, block_size, "qwen35 norm threads");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto plan = plan_qwen_threads(*threads.value, block_size);
    if (!plan.ok()) { return plan; }
    plan.value->work_items = rows;
    plan.value->warp_size = 32u;
    return plan;
}

// Register-cached model specialization (opt_plan/09 N3.1). See dsl_ops.h.
[[nodiscard]] QwenRmsNormKernel make_qwen35_rmsnorm_reg_kernel(
    QwenNormEpilogue epilogue, uint32_t dim, uint32_t block_size) noexcept {
    QwenRmsNormKernel kernel = [epilogue, dim, block_size](BufferFloat x, BufferFloat w, BufferFloat residual,
                                                           BufferFloat y, BufferFloat sum_out,
                                                           Var<uint> rows, Var<uint> arg_dim,
                                                           Var<float> eps) noexcept {
        set_block_size(block_size, 1u, 1u);
        set_warp_size(32u);
        set_name(epilogue == QwenNormEpilogue::residual ? "qwen35_rmsnorm_reg_residual"
                 : epilogue == QwenNormEpilogue::snapshot ? "qwen35_rmsnorm_reg_snapshot"
                                                          : "qwen35_rmsnorm_reg");
        // Both extents are compile-time: the element cache and the norm-weight
        // cache are constant-indexed, i.e. registers (no local-memory array).
        const uint32_t elems = block_size == 0u ? 0u : dim / block_size;
        const uint32_t warps = block_size == 0u ? 0u : block_size / 32u;
        Shared<float> partial{warps == 0u ? 1u : warps};
        Local<float> cache{elems == 0u ? 1u : elems};
        Local<float> wcache{elems == 0u ? 1u : elems};
        const auto row = qwen_linear_id() / block_size;
        const auto tid = thread_x();
        const auto lane = warp_lane_id();
        const auto wid = tid / 32u;
        // The runtime dim must be the compiled one: any other geometry is a
        // guarded no-op (the generic kernels serve it).
        $if (row < rows & arg_dim == dim) {
            const auto base = row * dim;
            Var<float> sumsq = def(0.0f);
            for (uint32_t j = 0u; j < elems; j++) {
                const auto i = tid + j * block_size;
                Var<float> v = x.read(base + i);
                if (epilogue == QwenNormEpilogue::residual) {
                    v += residual.read(base + i);
                    sum_out.write(base + i, v);
                } else if (epilogue == QwenNormEpilogue::snapshot) {
                    // Residual snapshot fused into the sum-of-squares pass: the
                    // ORIGINAL x is stored bit-exactly (no arithmetic on v), so a
                    // later consumer sees the same values as the generic kernel.
                    sum_out.write(base + i, v);
                }
                cache[j] = v;
                wcache[j] = w.read(i);
                sumsq += v * v;
            };
            sumsq = warp_active_sum(sumsq);
            $if (lane == 0u) { partial[wid] = sumsq; };
            sync_block();
            // Same four-partial order as the wide kernel, so the reduction (and
            // therefore the final rounding) is identical.
            sumsq = 0.0f;
            for (uint32_t p = 0u; p < warps; p++) {
                sumsq += partial.read(p);
            }
            const auto inv = rsqrt(sumsq / static_cast<float>(dim) + eps);
            // No second global pass: the value and the norm weight come from the
            // registers filled by the pass above.
            for (uint32_t j = 0u; j < elems; j++) {
                const auto i = tid + j * block_size;
                y.write(base + i, cache[j] * inv * wcache[j]);
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_rmsnorm_reg_dispatch(
    uint64_t rows, uint64_t dim, uint32_t block_size) {
    if (block_size == 0u || block_size % 32u != 0u || block_size > 1024u) {
        return Result<QwenLaunch1D>::failure(err_invalid(
            "register RMSNorm block size must be a multiple of 32 in [32,1024]"));
    }
    if (dim == 0u || dim % block_size != 0u) {
        return Result<QwenLaunch1D>::failure(err_invalid(
            "register RMSNorm requires dim % block_size == 0"));
    }
    if (dim > 0xFFFFFFFFull) {
        return Result<QwenLaunch1D>::failure(err_overflow("RMSNorm dim exceeds uint32"));
    }
    if (!checked_product_u32(rows, dim, "rmsnorm elements").ok()) {
        return Result<QwenLaunch1D>::failure(err_overflow("rmsnorm elements exceed uint32"));
    }
    const auto threads = checked_product_u32(rows, block_size, "rmsnorm threads");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto plan = plan_qwen_threads(*threads.value, block_size);
    if (!plan.ok()) { return plan; }
    plan.value->work_items = rows;
    plan.value->warp_size = 32u;
    return plan;
}

// ---------------------------------------------------------------------------
// 3/4. Packed GEMV / GEMM
// ---------------------------------------------------------------------------
namespace {

// Shared body: y[m,n] = dot(x[m,:], W[n,:]) (+residual). `epilogue` and
// `format` are compile-time. Used by both the warp GEMV and the thread GEMM.
[[nodiscard]] auto qwen_matmul_dot(const BufferVar<float> &x, const BufferVar<uint32_t> &w,
                                   const BufferVar<float> &scales,
                                   Var<uint> x_base, Var<uint> w_base, Var<uint> scale_base,
                                   Var<uint> k, Var<uint> group, Var<uint> begin, Var<uint> step,
                                   QwenWeightFormat format) noexcept {
    Var<float> acc = def(0.0f);
    $for (ki, begin, k, step) {
        const auto xv = x.read(x_base + ki);
        const auto wv = qwen_read_weight(w, scales, w_base + ki, scale_base, ki, group, format);
        acc += xv * wv;
    };
    return acc;
}

// Packed-word-oriented GEMV dot (warp lanes own whole packed words).
// Each lane owns packed word `begin` and strides by `step_words` words:
//   bf16: ki = 2*begin, K stride 64  -> one uint32, two bf16 halves;
//   fp8 : ki = 4*begin, K stride 128 -> one uint32, four e4m3 bytes;
//   q8_0: the same 4-codes-per-word geometry as fp8, four SIGNED int8 codes.
// Each packed word is decoded exactly once per lane iteration and the
// products accumulate in ascending K order. For fp8/q8_0 the caller guarantees
// K % 4 == 0 AND group % 4 == 0, so a quad never crosses a scale boundary:
// ONE scale load per word, no per-element group division.
// NOTE: this repartitions K across lanes (lane owns a contiguous quad/pair
// instead of single strided elements), so the warp reduction order -- and
// therefore the final rounding -- differs from qwen_matmul_dot. The FP32
// math per product is identical; results are NOT bit-identical to the
// strided path, only within accumulation-order noise.
[[nodiscard]] auto qwen_gemv_dot_packed(const BufferVar<float> &x, const BufferVar<uint32_t> &w,
                                        const BufferVar<float> &scales,
                                        Var<uint> x_base, Var<uint> w_base, Var<uint> scale_base,
                                        Var<uint> k, Var<uint> group,
                                        Var<uint> begin_word, Var<uint> step_words,
                                        QwenWeightFormat format) noexcept {
    Var<float> acc = def(0.0f);
    // `words` = k / pack (caller guarantees k % pack == 0). The word base
    // is computed once on the caller side (warp * words).
    const auto words = format == QwenWeightFormat::bf16 ? (k >> 1u)
                     : format == QwenWeightFormat::w4  ? (k >> 3u)
                                                       : (k >> 2u);
    // group % 4 == 0 and group > 0 are guaranteed for fp8 by the caller, so
    // ki / group == wi / (group / 4) for the quad's base index. W4 needs
    // group % 8 == 0 for the same reason (one word = 8 codes).
    const auto group_words = format == QwenWeightFormat::w4 ? (group >> 3u) : (group >> 2u);
    $for (wi, begin_word, words, step_words) {
        const auto word = w.read(w_base + wi);
        if (format == QwenWeightFormat::w4) {
            // One word = 8 signed int4 codes, one FP32 group scale: 8 products.
            const auto ki = wi * 8u;
            const auto scale = scales.read(scale_base + wi / group_words);
            const auto x0 = x.read(x_base + ki);
            const auto x1 = x.read(x_base + ki + 1u);
            const auto x2 = x.read(x_base + ki + 2u);
            const auto x3 = x.read(x_base + ki + 3u);
            const auto x4 = x.read(x_base + ki + 4u);
            const auto x5 = x.read(x_base + ki + 5u);
            const auto x6 = x.read(x_base + ki + 6u);
            const auto x7 = x.read(x_base + ki + 7u);
            acc += x0 * qwen_w4_to_f32(word & 0xFu) * scale;
            acc += x1 * qwen_w4_to_f32((word >> 4u) & 0xFu) * scale;
            acc += x2 * qwen_w4_to_f32((word >> 8u) & 0xFu) * scale;
            acc += x3 * qwen_w4_to_f32((word >> 12u) & 0xFu) * scale;
            acc += x4 * qwen_w4_to_f32((word >> 16u) & 0xFu) * scale;
            acc += x5 * qwen_w4_to_f32((word >> 20u) & 0xFu) * scale;
            acc += x6 * qwen_w4_to_f32((word >> 24u) & 0xFu) * scale;
            acc += x7 * qwen_w4_to_f32((word >> 28u) & 0xFu) * scale;
        } else if (format == QwenWeightFormat::bf16) {
            const auto ki = wi * 2u;
            const auto w0 = qwen_bf16_to_f32(word & 0xFFFFu);
            const auto w1 = qwen_bf16_to_f32((word >> 16u) & 0xFFFFu);
            acc += x.read(x_base + ki) * w0;
            acc += x.read(x_base + ki + 1u) * w1;
        } else {
            const auto ki = wi * 4u;
            const auto scale = scales.read(scale_base + wi / group_words);
            if (format == QwenWeightFormat::q8_0) {
                const auto w0 = qwen_q8_0_to_f32(word & 0xFFu) * scale;
                const auto w1 = qwen_q8_0_to_f32((word >> 8u) & 0xFFu) * scale;
                const auto w2 = qwen_q8_0_to_f32((word >> 16u) & 0xFFu) * scale;
                const auto w3 = qwen_q8_0_to_f32((word >> 24u) & 0xFFu) * scale;
                acc += x.read(x_base + ki) * w0;
                acc += x.read(x_base + ki + 1u) * w1;
                acc += x.read(x_base + ki + 2u) * w2;
                acc += x.read(x_base + ki + 3u) * w3;
            } else {
                const auto w0 = qwen_fp8_e4m3_to_f32(word & 0xFFu) * scale;
                const auto w1 = qwen_fp8_e4m3_to_f32((word >> 8u) & 0xFFu) * scale;
                const auto w2 = qwen_fp8_e4m3_to_f32((word >> 16u) & 0xFFu) * scale;
                const auto w3 = qwen_fp8_e4m3_to_f32((word >> 24u) & 0xFFu) * scale;
                acc += x.read(x_base + ki) * w0;
                acc += x.read(x_base + ki + 1u) * w1;
                acc += x.read(x_base + ki + 2u) * w2;
                acc += x.read(x_base + ki + 3u) * w3;
            }
        }
    };
    return acc;
}

// Fast-path eligibility (host-computed shape facts, uniform across the
// warp): K divides the pack width, and for the scaled formats the group divides
// the quad.
[[nodiscard]] auto qwen_gemv_packed_ok(Var<uint> k, Var<uint> group,
                                       QwenWeightFormat format) noexcept {
    if (format == QwenWeightFormat::bf16) {
        return (k & 1u) == 0u;
    }
    if (format == QwenWeightFormat::w4) {
        return (k & 7u) == 0u & (group & 7u) == 0u & group > 0u;
    }
    return (k & 3u) == 0u & (group & 3u) == 0u & group > 0u;
}

}// namespace

[[nodiscard]] QwenGemvKernel make_qwen_gemv_kernel(QwenWeightFormat format,
                                                   QwenMatmulEpilogue epilogue) noexcept {
    QwenGemvKernel kernel = [format, epilogue](BufferFloat x, BufferUInt w, BufferFloat scales,
                                               BufferFloat residual, BufferFloat y,
                                               Var<uint> m, Var<uint> n, Var<uint> k, Var<uint> group) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(luisa::string{"qwen_gemv"} + qwen_format_name_suffix(format));
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < n & k > 0u & m > 0u) {
            const auto w_base = warp * k;
            Var<uint> scale_base = def(0u);
            if (format != QwenWeightFormat::bf16) {
                scale_base = warp * ((k + group - 1u) / group);
            }
            // Each lane accumulates its K subset for every m row, then one
            // warp reduction per row. Fast path: lane owns whole packed
            // words (one word read + decode per pair/quad, one fp8 scale
            // load per quad, no per-element group division); lanes read
            // consecutive words (coalesced along K). Ragged K (or fp8 group
            // not divisible by 4) falls back to the legacy per-element
            // strided loop. The $if condition is warp-uniform, so
            // warp_active_sum stays block-uniform on both paths. The two
            // paths partition K differently -> rounding differs (see
            // qwen_gemv_dot_packed).
            $for (mi, m) {
                Var<float> acc = def(0.0f);
                $if (qwen_gemv_packed_ok(k, group, format)) {
                    const auto word_base = format == QwenWeightFormat::bf16 ? w_base / 2u
                                         : format == QwenWeightFormat::w4  ? w_base / 8u
                                                                           : w_base / 4u;
                    acc = qwen_gemv_dot_packed(x, w, scales, mi * k, word_base, scale_base, k, group,
                                               lane, 32u, format);
                } $else {
                    acc = qwen_matmul_dot(x, w, scales, mi * k, w_base, scale_base, k, group,
                                          lane, 32u, format);
                };
                acc = warp_active_sum(acc);
                $if (lane == 0u) {
                    const auto idx = mi * n + warp;
                    Var<float> v = acc;
                    if (epilogue == QwenMatmulEpilogue::residual) {
                        v += residual.read(idx);
                    }
                    y.write(idx, v);
                };
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gemv_dispatch(uint64_t m, uint64_t n, uint64_t k) {
    if (m == 0u || m > kQwenGemvMaxM) {
        return Result<QwenLaunch1D>::failure(err_invalid("gemv M must be in [1,4]"));
    }
    const auto xk = checked_product_u32(m, k, "gemv x");
    if (!xk.ok()) { return Result<QwenLaunch1D>::failure(xk.status); }
    const auto wn = checked_product_u32(n, k, "gemv weights");
    if (!wn.ok()) { return Result<QwenLaunch1D>::failure(wn.status); }
    const auto yn = checked_product_u32(m, n, "gemv output");
    if (!yn.ok()) { return Result<QwenLaunch1D>::failure(yn.status); }
    return plan_warp_rows(n, 128u);
}

namespace {

// Decode-only Qwen shape specialization. `words` is a multiple of 128, so
// every lane executes four valid, independent streams without tail guards.
// Keeping the streams in separate accumulators also breaks the single FMA
// dependency chain of the generic packed-word loop.
[[nodiscard]] auto qwen35_decode_dot(const BufferVar<float> &x, const BufferVar<uint32_t> &w,
                                     const BufferVar<float> &scales, Shared<float> &lut,
                                     Var<uint> w_base, Var<uint> scale_base,
                                     Var<uint> k, Var<uint> group,
                                     Var<uint> lane, QwenWeightFormat format) noexcept {
    Var<float> a0 = def(0.0f), a1 = def(0.0f), a2 = def(0.0f), a3 = def(0.0f);
    if (format == QwenWeightFormat::w4) {
        // W4A16 decode: one lane reads ONE uint32 = 8 signed int4 codes and does
        // 32 lanes * 8 = 256 K elements per warp iteration. The 8 products of a
        // word are split over the SAME four independent accumulators the 8-bit
        // realizations use (2 products each), so the FMA dependency chain stays
        // 4-wide. The scale is hoisted per word (one load per 8 weights rather
        // than per element); a group of 128 spans 16 words, so lanes that stride
        // 32 words see one scale per word, which is the maximum hoist available
        // at this repartition. `$for` guards the K tail: words = K/8 need not be
        // a multiple of 32 (K = 2560 -> 320 words, K = 9216 -> 1152, N any).
        const auto words = k >> 3u;
        const auto group_words = group >> 3u;
        // `w_base` is an ELEMENT index (warp * k); a word holds 8 codes.
        const auto word_base = w_base / 8u;
        $for (wi, lane, words, 32u) {
            const auto word = w.read(word_base + wi);
            const auto scale = scales.read(scale_base + wi / group_words);
            const auto ki = wi * 8u;
            const auto x0 = x.read(ki);
            const auto x1 = x.read(ki + 1u);
            const auto x2 = x.read(ki + 2u);
            const auto x3 = x.read(ki + 3u);
            const auto x4 = x.read(ki + 4u);
            const auto x5 = x.read(ki + 5u);
            const auto x6 = x.read(ki + 6u);
            const auto x7 = x.read(ki + 7u);
            a0 += x0 * qwen_w4_to_f32(word & 0xFu) * scale;
            a0 += x1 * qwen_w4_to_f32((word >> 4u) & 0xFu) * scale;
            a1 += x2 * qwen_w4_to_f32((word >> 8u) & 0xFu) * scale;
            a1 += x3 * qwen_w4_to_f32((word >> 12u) & 0xFu) * scale;
            a2 += x4 * qwen_w4_to_f32((word >> 16u) & 0xFu) * scale;
            a2 += x5 * qwen_w4_to_f32((word >> 20u) & 0xFu) * scale;
            a3 += x6 * qwen_w4_to_f32((word >> 24u) & 0xFu) * scale;
            a3 += x7 * qwen_w4_to_f32((word >> 28u) & 0xFu) * scale;
        };
        return (a0 + a1) + (a2 + a3);
    }
    const auto words = format == QwenWeightFormat::bf16 ? (k >> 1u) : (k >> 2u);
    const auto group_words = group >> 2u;
    const auto word_base = format == QwenWeightFormat::bf16 ? w_base / 2u : w_base / 4u;
    const auto accumulate = [&](Var<uint> wi, Var<float> &acc) {
        const auto word = w.read(word_base + wi);
        if (format == QwenWeightFormat::bf16) {
            const auto ki = wi * 2u;
            acc += x.read(ki) * qwen_bf16_to_f32(word & 0xFFFFu);
            acc += x.read(ki + 1u) * qwen_bf16_to_f32(word >> 16u);
        } else {
            const auto ki = wi * 4u;
            const auto scale = scales.read(scale_base + wi / group_words);
            if (format == QwenWeightFormat::q8_0) {
                // Q8_0: exact signed-int8 decode * one FP32 block scale per 8
                // words. No shared LUT (int8 -> fp32 is two integer ops), so
                // this path never touches `lut` and needs no barrier.
                acc += x.read(ki) * qwen_q8_0_to_f32(word & 0xFFu) * scale;
                acc += x.read(ki + 1u) * qwen_q8_0_to_f32((word >> 8u) & 0xFFu) * scale;
                acc += x.read(ki + 2u) * qwen_q8_0_to_f32((word >> 16u) & 0xFFu) * scale;
                acc += x.read(ki + 3u) * qwen_q8_0_to_f32((word >> 24u) & 0xFFu) * scale;
            } else {
                acc += x.read(ki) * lut.read(word & 0xFFu) * scale;
                acc += x.read(ki + 1u) * lut.read((word >> 8u) & 0xFFu) * scale;
                acc += x.read(ki + 2u) * lut.read((word >> 16u) & 0xFFu) * scale;
                acc += x.read(ki + 3u) * lut.read((word >> 24u) & 0xFFu) * scale;
            }
        }
    };
    $for (wi, lane, words, 128u) {
        accumulate(wi, a0);
        accumulate(wi + 32u, a1);
        accumulate(wi + 64u, a2);
        accumulate(wi + 96u, a3);
    };
    return (a0 + a1) + (a2 + a3);
}

}// namespace

[[nodiscard]] QwenGemvKernel make_qwen35_decode_gemv_kernel(
    QwenWeightFormat format, QwenMatmulEpilogue epilogue) noexcept {
    QwenGemvKernel kernel = [format, epilogue](BufferFloat x, BufferUInt w, BufferFloat scales,
                                               BufferFloat residual, BufferFloat y,
                                               Var<uint> m, Var<uint> n, Var<uint> k,
                                               Var<uint> group) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(luisa::string{"qwen35_decode_gemv"} + qwen_format_name_suffix(format));
        Shared<float> lut{format == QwenWeightFormat::fp8_e4m3 ? 256u : 1u};
        if (format == QwenWeightFormat::fp8_e4m3) {
            // Two entries per thread. This exact conversion is paid once per
            // block, not once per packed weight; all consumers follow the
            // block barrier. Random LUT reads remain in 1 KiB of shared memory.
            $for (i, thread_x(), 256u, 128u) {
                lut[i] = qwen_fp8_e4m3_to_f32(i);
            };
            sync_block();
        }
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < n & m == 1u & k > 0u) {
            const auto w_base = warp * k;
            Var<uint> scale_base = def(0u);
            if (format != QwenWeightFormat::bf16) {
                scale_base = warp * ((k + group - 1u) / group);
            }
            auto acc = qwen35_decode_dot(x, w, scales, lut, w_base, scale_base,
                                         k, group, lane, format);
            acc = warp_active_sum(acc);
            $if (lane == 0u) {
                Var<float> v = acc;
                if (epilogue == QwenMatmulEpilogue::residual) {
                    v += residual.read(warp);
                }
                y.write(warp, v);
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_decode_gemv_dispatch(
    uint64_t n, uint64_t k, QwenWeightFormat format, uint32_t group) {
    if (format == QwenWeightFormat::w4) {
        // The W4 realization walks one word per lane with a stride of 32 words
        // and guards the tail, so only whole words and whole scale groups are
        // required (K = 2560/4096/9216, N any).
        if (auto st = validate_qwen_w4_geometry(k, group); !st.ok()) {
            return Result<QwenLaunch1D>::failure(std::move(st));
        }
        const auto wn = checked_product_u32(n, k, "decode GEMV weights");
        if (!wn.ok()) { return Result<QwenLaunch1D>::failure(wn.status); }
        return plan_warp_rows(n, 128u);
    }
    const uint64_t pack = format == QwenWeightFormat::bf16 ? 2u : 4u;
    const uint64_t words_per_launch = 128u * pack;
    if (k == 0u || k % words_per_launch != 0u) {
        return Result<QwenLaunch1D>::failure(err_invalid(
            "qwen35 decode GEMV requires K divisible by 128 packed words"));
    }
      (void)group;
      return plan_qwen_gemv_dispatch(1u, n, k);
  }

  // ---------------------------------------------------------------------------
  // 3a-bis. MERGED decode input projections (opt_plan/03 section 3 + D5).
  //
  // One warp per merged OUTPUT ROW, exactly like the two realizations it replaces,
  // and the SAME dot body per row group: the projection rows take the realization
  // the unmerged decode route compiles for this format, the GDN alpha/beta rows take
  // the fused gate kernel's realization `qwen35_decode_dot` + its epilogue. The
  // per-output FP32 accumulation order is therefore unchanged (same lane partition,
  // same in-word product order, same four-accumulator tree, one warp_active_sum), so
  // merging cannot move a token: the decode logits of a merged session are
  // bit-identical to the unmerged one. Nothing is staged: each warp writes its
  // scalar straight into the row's real destination buffer.
  // ---------------------------------------------------------------------------
  namespace {

  // Fail-closed launch/geometry check shared by both merged plans. `k` must obey the
  // decode GEMV packed contract of `format` (the dispatch reuses the same plan), and
  // no segment may be empty.
  [[nodiscard]] Status validate_merged_rows(uint64_t rows, uint64_t k, QwenWeightFormat format,
                                            uint32_t group) {
      if (format == QwenWeightFormat::w4) {
            return Status::error(StatusCode::unsupported,
                                 "the merged decode projection has no W4 realization (only the "
                                 "BF16/FP8/Q8_0 decode GEMV was measured); keep W4 on its own pack");
      }
      if (rows == 0u || rows > 0xFFFFFFFFull) {
          return err_invalid("merged decode projection row count is outside the uint32 envelope");
      }
      auto plan = plan_qwen35_decode_gemv_dispatch(rows, k, format, group);
      if (!plan.ok()) { return plan.status; }
      return Status::success();
  }

  }// namespace

  [[nodiscard]] QwenMergedGdnGemvKernel make_qwen35_merged_gdn_gemv_kernel(
      QwenWeightFormat format) noexcept {
      // The projection-row dot realization: BF16 keeps the generic packed pair loop
      // (that is what the engine compiles for BF16 decode projections), every scaled
      // format uses the four-stream specialization with the shared E4M3 LUT.
      const bool specialized = format != QwenWeightFormat::bf16;
      QwenMergedGdnGemvKernel kernel = [format, specialized](
          BufferFloat x, BufferUInt w, BufferFloat scales, BufferFloat qkv_out, BufferFloat z_out,
          BufferFloat ssm_a, BufferFloat dt_bias, BufferFloat beta_out, BufferFloat g_out,
          Var<uint> k, Var<uint> group, Var<uint> qkv_rows, Var<uint> z_rows, Var<uint> gate_heads,
          Var<float> softplus_threshold) noexcept {
          set_block_size(128u, 1u, 1u);
          set_warp_size(32u);
          set_name(luisa::string{"qwen35_merged_gdn_gemv"} + qwen_format_name_suffix(format));
          Shared<float> lut{format == QwenWeightFormat::fp8_e4m3 ? 256u : 1u};
          if (format == QwenWeightFormat::fp8_e4m3) {
              $for (i, thread_x(), 256u, 128u) { lut[i] = qwen_fp8_e4m3_to_f32(i); };
              sync_block();
          }
          const auto warp = qwen_linear_id() / 32u;
          const auto lane = warp_lane_id();
          const auto gate_begin = qkv_rows + z_rows;
          const auto rows = gate_begin + gate_heads * 2u;
          $if (warp < rows & k > 0u) {
              const auto w_base = warp * k;
              Var<uint> scale_base = def(0u);
              if (format != QwenWeightFormat::bf16) {
                  scale_base = warp * ((k + group - 1u) / group);
              }
              Var<float> acc = def(0.0f);
              $if (warp < gate_begin) {
                  // Projection row: the decode projection realization of this format,
                  // including the ragged-K fallback the generic kernel would take.
                  if (specialized) {
                      acc = qwen35_decode_dot(x, w, scales, lut, w_base, scale_base, k, group,
                                              lane, format);
                  } else {
                      const auto word_base = format == QwenWeightFormat::bf16 ? w_base / 2u
                                                                            : w_base / 4u;
                      $if (qwen_gemv_packed_ok(k, group, format)) {
                          acc = qwen_gemv_dot_packed(x, w, scales, 0u, word_base, scale_base, k,
                                                     group, lane, 32u, format);
                      } $else {
                          acc = qwen_matmul_dot(x, w, scales, 0u, w_base, scale_base, k, group,
                                                lane, 32u, format);
                      };
                  }
              } $else {
                  // GDN alpha/beta row: the fused gate kernel's dot (format-agnostic
                  // four-stream realization), so its result is bit-identical to
                  // make_qwen_gdn_decode_gates_kernel on the same source row.
                  acc = qwen35_decode_dot(x, w, scales, lut, w_base, scale_base, k, group,
                                          lane, format);
              };
              acc = warp_active_sum(acc);
              $if (lane == 0u) {
                  // Segment-aware destinations: every warp stores directly into the
                  // buffer its row belongs to (no intermediate, no extra pass).
                  $if (warp < qkv_rows) {
                      qkv_out.write(warp, acc);
                  };
                  $if (warp >= qkv_rows & warp < gate_begin) {
                      z_out.write(warp - qkv_rows, acc);
                  };
                  $if (warp >= gate_begin) {
                      const auto g = warp - gate_begin;
                      // alpha columns [0, gate_heads): g = ssm_a*softplus(a + dt_bias)
                      $if (g < gate_heads) {
                          const auto av = acc + dt_bias.read(g);
                          Var<float> sp = def(0.0f);
                          $if (av > softplus_threshold) {
                              sp = av;
                          } $else {
                              sp = log(1.0f + exp(av));
                          };
                          g_out.write(g, ssm_a.read(g) * sp);
                      };
                      // beta columns [gate_heads, 2*gate_heads): beta = sigmoid(b)
                      $if (g >= gate_heads) {
                          beta_out.write(g - gate_heads, qwen_sigmoid(acc));
                      };
                  };
              };
          };
      };
      return kernel;
  }

  [[nodiscard]] Result<QwenLaunch1D> plan_qwen35_merged_gdn_gemv_dispatch(
      uint64_t qkv_rows, uint64_t z_rows, uint64_t gate_heads, uint64_t k,
      QwenWeightFormat format, uint32_t group) {
      if (qkv_rows == 0u || z_rows == 0u || gate_heads == 0u ||
          qkv_rows > 0xFFFFFFFFull || z_rows > 0xFFFFFFFFull || gate_heads > 0xFFFFFFFFull) {
          return Result<QwenLaunch1D>::failure(
              err_invalid("merged GDN projection requires three non-empty segments"));
      }
      const uint64_t rows = qkv_rows + z_rows + 2ull * gate_heads;
      if (auto s = validate_merged_rows(rows, k, format, group); !s.ok()) {
          return Result<QwenLaunch1D>::failure(std::move(s));
      }
      return plan_warp_rows(rows, 128u);
  }

  [[nodiscard]] QwenMergedAttnGemvKernel make_qwen35_merged_attn_gemv_kernel(
      QwenWeightFormat format) noexcept {
      const bool specialized = format != QwenWeightFormat::bf16;
      QwenMergedAttnGemvKernel kernel = [format, specialized](
          BufferFloat x, BufferUInt w, BufferFloat scales, BufferFloat q_out, BufferFloat k_out,
          BufferFloat v_out, Var<uint> k, Var<uint> group, Var<uint> q_rows,
          Var<uint> kv_rows) noexcept {
          set_block_size(128u, 1u, 1u);
          set_warp_size(32u);
          set_name(luisa::string{"qwen35_merged_attn_gemv"} + qwen_format_name_suffix(format));
          Shared<float> lut{format == QwenWeightFormat::fp8_e4m3 ? 256u : 1u};
          if (format == QwenWeightFormat::fp8_e4m3) {
              $for (i, thread_x(), 256u, 128u) { lut[i] = qwen_fp8_e4m3_to_f32(i); };
              sync_block();
          }
          const auto warp = qwen_linear_id() / 32u;
          const auto lane = warp_lane_id();
          const auto k_begin = q_rows;
          const auto v_begin = q_rows + kv_rows;
          const auto rows = v_begin + kv_rows;
          $if (warp < rows & k > 0u) {
              const auto w_base = warp * k;
              Var<uint> scale_base = def(0u);
              if (format != QwenWeightFormat::bf16) {
                  scale_base = warp * ((k + group - 1u) / group);
              }
              Var<float> acc = def(0.0f);
              if (specialized) {
                  acc = qwen35_decode_dot(x, w, scales, lut, w_base, scale_base, k, group,
                                          lane, format);
              } else {
                  const auto word_base = format == QwenWeightFormat::bf16 ? w_base / 2u
                                                                        : w_base / 4u;
                  $if (qwen_gemv_packed_ok(k, group, format)) {
                      acc = qwen_gemv_dot_packed(x, w, scales, 0u, word_base, scale_base, k,
                                                 group, lane, 32u, format);
                  } $else {
                      acc = qwen_matmul_dot(x, w, scales, 0u, w_base, scale_base, k, group,
                                            lane, 32u, format);
                  };
              }
              acc = warp_active_sum(acc);
              $if (lane == 0u) {
                  // Segment-aware destinations: Q rows then K rows then V rows.
                  $if (warp < k_begin) {
                      q_out.write(warp, acc);
                  };
                  $if (warp >= k_begin & warp < v_begin) {
                      k_out.write(warp - k_begin, acc);
                  };
                  $if (warp >= v_begin) {
                      v_out.write(warp - v_begin, acc);
                  };
              };
          };
      };
      return kernel;
  }

  [[nodiscard]] Result<QwenLaunch1D> plan_qwen35_merged_attn_gemv_dispatch(
      uint64_t q_rows, uint64_t kv_rows, uint64_t k, QwenWeightFormat format, uint32_t group) {
      if (q_rows == 0u || kv_rows == 0u || q_rows > 0xFFFFFFFFull || kv_rows > 0xFFFFFFFFull) {
          return Result<QwenLaunch1D>::failure(
              err_invalid("merged attention projection requires three non-empty segments"));
      }
      const uint64_t rows = q_rows + 2ull * kv_rows;
      if (auto s = validate_merged_rows(rows, k, format, group); !s.ok()) {
          return Result<QwenLaunch1D>::failure(std::move(s));
      }
      return plan_warp_rows(rows, 128u);
  }


// ---------------------------------------------------------------------------
// 3b. Vectorized 128-bit decode GEMV (opt_plan/09 N2). See dsl_ops.h.
// ---------------------------------------------------------------------------
namespace {
// Fail-closed geometry of every 128-bit decode kernel: K must be a whole number
// of 32-uint4 lane strides so the inner loop has no tail predicate, and (for the
// scaled formats) a uint4 must never cross a quantisation group.
[[nodiscard]] Status validate_packed_decode_geometry(uint64_t k, QwenWeightFormat format,
                                                     uint32_t group) noexcept {
    const uint64_t elems = qwen_packed_vec_elements(format);
    const uint64_t lane_stride = 32u * elems;
    if (k == 0u || k % lane_stride != 0u) {
        return err_invalid(format == QwenWeightFormat::bf16
                               ? "packed decode requires K % 256 == 0 (32 lanes x 8 BF16)"
                               : "packed decode requires K % 512 == 0 (32 lanes x 16 codes)");
    }
    if (format == QwenWeightFormat::w4) {
        // No 128-bit W4 realization is implemented: one uint4 would be 32 codes
        // and the 32-lane stride 1024 K elements, which no model shape satisfies
        // (K = 2560 is not a multiple of 1024) and which has not been measured.
        // Refusing here keeps the option honest instead of pretending it works.
        return err_invalid("no 128-bit W4 decode realization (use the 32-bit decode GEMV)");
    }
    if (format == QwenWeightFormat::q8_0) {
        // One uint4 = 16 int8 codes; a 32-code GGUF block spans exactly two
        // uint4s, so the group must be the block size and divide K.
        if (group != kQwenQ8BlockSize) {
            return err_invalid("packed Q8_0 decode requires the 32-element GGUF block group");
        }
        if (k % group != 0u) {
            return err_invalid("packed Q8_0 decode requires the block group to divide K");
        }
        return Status::success();
    }
    if (format == QwenWeightFormat::fp8_e4m3) {
        if (group == 0u || group % 16u != 0u) {
            return err_invalid("packed decode requires an FP8 group that is a multiple of 16");
        }
        if (k % group != 0u) {
            return err_invalid("packed decode requires the FP8 group to divide K");
        }
    }
    return Status::success();
}
}// namespace

[[nodiscard]] QwenGemvPackedKernel make_qwen35_decode_gemv_packed_kernel(
    QwenWeightFormat format, QwenMatmulEpilogue epilogue) noexcept {
    const bool fp8 = format == QwenWeightFormat::fp8_e4m3;
    const bool q8 = format == QwenWeightFormat::q8_0;
    const bool scaled = fp8 || q8;
    QwenGemvPackedKernel kernel = [fp8, q8, scaled, epilogue, format](
                                     BufferFloat4 x, BufferUInt4 w, BufferFloat scales,
                                     BufferFloat residual, BufferFloat y,
                                     Var<uint> n, Var<uint> k, Var<uint> group,
                                     Var<uint> vecs) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(luisa::string{"qwen35_decode_gemv_packed"} + qwen_format_name_suffix(format));
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < n & k > 0u) {
            const auto w_base = warp * vecs;// uint4 units
            Var<uint> scale_base = def(0u);
            if (scaled) { scale_base = warp * ((k + group - 1u) / group); }
            Var<uint> group_vecs = def(1u);
            if (scaled) { group_vecs = group >> 4u; }
            // Four accumulators, one per packed word of the uint4: the final
            // (a0+a1)+(a2+a3) tree is the same shape the 32-bit decode kernel
            // uses, so only the per-accumulator term SET differs.
            Var<float> a0 = def(0.0f), a1 = def(0.0f), a2 = def(0.0f), a3 = def(0.0f);
            $for (vi, lane, vecs, 32u) {
                const auto wv = w.read(w_base + vi);
                if (scaled) {
                    const auto s = scales.read(scale_base + vi / group_vecs);
                    const auto v0 = x.read(vi * 4u);
                    const auto v1 = x.read(vi * 4u + 1u);
                    const auto v2 = x.read(vi * 4u + 2u);
                    const auto v3 = x.read(vi * 4u + 3u);
                    a0 += v0.x * qwen_scaled_byte_to_f32(wv.x & 0xFFu, format) * s;
                    a0 += v0.y * qwen_scaled_byte_to_f32((wv.x >> 8u) & 0xFFu, format) * s;
                    a0 += v0.z * qwen_scaled_byte_to_f32((wv.x >> 16u) & 0xFFu, format) * s;
                    a0 += v0.w * qwen_scaled_byte_to_f32((wv.x >> 24u) & 0xFFu, format) * s;
                    a1 += v1.x * qwen_scaled_byte_to_f32(wv.y & 0xFFu, format) * s;
                    a1 += v1.y * qwen_scaled_byte_to_f32((wv.y >> 8u) & 0xFFu, format) * s;
                    a1 += v1.z * qwen_scaled_byte_to_f32((wv.y >> 16u) & 0xFFu, format) * s;
                    a1 += v1.w * qwen_scaled_byte_to_f32((wv.y >> 24u) & 0xFFu, format) * s;
                    a2 += v2.x * qwen_scaled_byte_to_f32(wv.z & 0xFFu, format) * s;
                    a2 += v2.y * qwen_scaled_byte_to_f32((wv.z >> 8u) & 0xFFu, format) * s;
                    a2 += v2.z * qwen_scaled_byte_to_f32((wv.z >> 16u) & 0xFFu, format) * s;
                    a2 += v2.w * qwen_scaled_byte_to_f32((wv.z >> 24u) & 0xFFu, format) * s;
                    a3 += v3.x * qwen_scaled_byte_to_f32(wv.w & 0xFFu, format) * s;
                    a3 += v3.y * qwen_scaled_byte_to_f32((wv.w >> 8u) & 0xFFu, format) * s;
                    a3 += v3.z * qwen_scaled_byte_to_f32((wv.w >> 16u) & 0xFFu, format) * s;
                    a3 += v3.w * qwen_scaled_byte_to_f32((wv.w >> 24u) & 0xFFu, format) * s;
                } else {
                    const auto v0 = x.read(vi * 2u);
                    const auto v1 = x.read(vi * 2u + 1u);
                    a0 += v0.x * qwen_bf16_to_f32(wv.x & 0xFFFFu);
                    a0 += v0.y * qwen_bf16_to_f32((wv.x >> 16u) & 0xFFFFu);
                    a1 += v0.z * qwen_bf16_to_f32(wv.y & 0xFFFFu);
                    a1 += v0.w * qwen_bf16_to_f32((wv.y >> 16u) & 0xFFFFu);
                    a2 += v1.x * qwen_bf16_to_f32(wv.z & 0xFFFFu);
                    a2 += v1.y * qwen_bf16_to_f32((wv.z >> 16u) & 0xFFFFu);
                    a3 += v1.z * qwen_bf16_to_f32(wv.w & 0xFFFFu);
                    a3 += v1.w * qwen_bf16_to_f32((wv.w >> 16u) & 0xFFFFu);
                }
            };
            const auto acc = warp_active_sum((a0 + a1) + (a2 + a3));
            $if (lane == 0u) {
                Var<float> v = acc;
                if (epilogue == QwenMatmulEpilogue::residual) {
                    v += residual.read(warp);
                }
                y.write(warp, v);
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_decode_gemv_packed_dispatch(
    uint64_t n, uint64_t k, QwenWeightFormat format, uint32_t group) {
    if (auto s = validate_packed_decode_geometry(k, format, group); !s.ok()) {
        return Result<QwenLaunch1D>::failure(std::move(s));
    }
    const auto wn = checked_product_u32(n, k, "packed decode GEMV weights");
    if (!wn.ok()) { return Result<QwenLaunch1D>::failure(wn.status); }
    return plan_warp_rows(n, 128u);
}

[[nodiscard]] QwenGemmKernel make_qwen_gemm_kernel(QwenWeightFormat format,
                                                   QwenMatmulEpilogue epilogue) noexcept {
    QwenGemmKernel kernel = [format, epilogue](BufferFloat x, BufferUInt w, BufferFloat scales,
                                               BufferFloat residual, BufferFloat y,
                                               Var<uint> m, Var<uint> n, Var<uint> k, Var<uint> group,
                                               Var<uint> padded_cols) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name(luisa::string{"qwen_gemm"} + qwen_format_name_suffix(format));
        const auto linear = qwen_linear_id();
        const auto row = linear / padded_cols;
        const auto col = linear % padded_cols;
        $if (row < m & col < n & k > 0u) {
            Var<uint> scale_base = def(0u);
            if (format != QwenWeightFormat::bf16) {
                scale_base = col * ((k + group - 1u) / group);
            }
            auto acc = qwen_matmul_dot(x, w, scales, row * k, col * k, scale_base, k, group,
                                       0u, 1u, format);
            const auto idx = row * n + col;
            Var<float> v = acc;
            if (epilogue == QwenMatmulEpilogue::residual) {
                v += residual.read(idx);
            }
            y.write(idx, v);
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch2D> plan_qwen_gemm_dispatch(uint64_t m, uint64_t n, uint64_t k) {
    const auto xk = checked_product_u32(m, k, "gemm x");
    if (!xk.ok()) { return Result<QwenLaunch2D>::failure(xk.status); }
    const auto wn = checked_product_u32(n, k, "gemm weights");
    if (!wn.ok()) { return Result<QwenLaunch2D>::failure(wn.status); }
    const auto yn = checked_product_u32(m, n, "gemm output");
    if (!yn.ok()) { return Result<QwenLaunch2D>::failure(yn.status); }
    return plan_qwen_threads_2d(m, n);
}

// ---------------------------------------------------------------------------
// 4b. Weight-reusing prefill GEMM (opt_plan/02 section 3). See dsl_ops.h.
// ---------------------------------------------------------------------------
[[nodiscard]] Status validate_qwen35_gemm_mt_config(const QwenGemmMtConfig &cfg) noexcept {
    if (cfg.mt == 0u || cfg.mt > 32u) {
        return err_invalid("mt GEMM: mt must be in [1,32] (live FP32 accumulator set)");
    }
    if (cfg.warps == 0u || cfg.warps > 8u) {
        return err_invalid("mt GEMM: warps must be in [1,8] (one 32-lane warp per output column)");
    }
    return Status::success();
}

[[nodiscard]] Status select_qwen35_gemm_mt_config(uint32_t warp_size, QwenGemmMtConfig &cfg) noexcept {
    if (warp_size != 32u) {
        return Status::error(StatusCode::unsupported,
                             "mt prefill GEMM pins warp size 32",
                             luisa::string{"device reports warp size "} +
                                 luisa::string{std::to_string(warp_size).c_str()});
    }
    QwenGemmMtConfig candidate{};
    if (auto s = validate_qwen35_gemm_mt_config(candidate); !s.ok()) { return s; }
    cfg = candidate;
    return Status::success();
}

[[nodiscard]] QwenGemmMtKernel make_qwen35_gemm_mt_kernel(
    const QwenGemmMtConfig &cfg, QwenWeightFormat format,
    QwenMatmulEpilogue epilogue) noexcept {
    const uint32_t mt = cfg.mt;
    const uint32_t warps = cfg.warps;
    const uint32_t block = 32u * warps;
    const uint32_t pack = qwen_weight_pack(format);
    const bool fp8 = format == QwenWeightFormat::fp8_e4m3;
    const bool q8 = format == QwenWeightFormat::q8_0;
    const bool w4 = format == QwenWeightFormat::w4;
    const bool scaled = fp8 || q8 || w4;
    QwenGemmMtKernel kernel = [=](BufferFloat x, BufferUInt w, BufferFloat scales,
                                 BufferFloat residual, BufferFloat y,
                                 Var<uint> m, Var<uint> n, Var<uint> k, Var<uint> group,
                                 Var<uint> column_blocks) noexcept {
        set_block_size(block, 1u, 1u);
        set_warp_size(32u);
        set_name(luisa::string{"qwen35_gemm_mt"} + qwen_format_name_suffix(format));
        // fp8: one shared 256-entry table of the EXACT E4M3->FP32 values, built
        // once per block, exactly like make_qwen35_decode_gemv_kernel: it keeps the
        // subnormal normalization loop out of the inner loop. MEASURED: on this box it
        // is worth ~0-2 % (the fp8 arm is bound by its per-row activation loads, not by
        // the decode; without the table fp8 measured 93.5-104.9 GB/s and with it
        // 95.2-107.0 GB/s), so the identical VALUES -- one shared conversion function --
        // are the reason to keep it.
        // (same conversion function), only its cost changes.
        // Q8_0 needs no table: the int8 decode is two integer ops, so the
        // shared LUT (and its barrier) exist only for the FP8 realization.
        Shared<float> lut{fp8 ? 256u : 1u};
        if (fp8) {
            $for (i, thread_x(), 256u, block) { lut[i] = qwen_fp8_e4m3_to_f32(i); };
            sync_block();
        }
        const auto lane = warp_lane_id();
        const auto warp = thread_x() / 32u;// one output column per warp
        // 2D grid (column_blocks, stripes) of 32*warps-thread blocks, one warp
        // per output column: the linear id is stripe * column_blocks + column
        // block, i.e. the column block is the fast axis so consecutive blocks
        // read neighboring weight rows and the x tile they share stays in L1/L2.
        const auto block_id = qwen_linear_id() / block;
        const auto cb = max(column_blocks, 1u);
        const auto col = (block_id % cb) * warps + warp;
        const auto row0 = (block_id / cb) * mt;
        // Warp-uniform guard: col depends on `warp`, row0/k on the block only,
        // so every lane of a participating warp enters (the butterfly reduction below
        // requires it).
        $if (col < n & row0 < m & k > 0u) {
            const auto words = k / pack;// planner guarantees k % pack == 0
            const auto w_base = col * words;
            const auto group_words = group / pack;// planner: group % pack == 0, > 0
            const auto scale_base = col * ((k + group - 1u) / group);
            // Register state: mt FP32 accumulators (compile-time extent/index).
            Local<float> acc{mt};
            Local<float> wv{pack};
            for (uint32_t r = 0u; r < mt; r++) { acc[r] = 0.0f; }
            // Lane l owns packed words l, l+32, ... of this output row: one
            // coalesced 128 B read per warp instruction, decoded ONCE and then
            // reused by all mt rows (the whole point of the family).
            $for (wi, lane, words, 32u) {
                const auto word = w.read(w_base + wi);
                if (scaled) {
                    // One scale load per packed word: the group is a whole
                    // number of words for every scaled format (fp8 64 = 16
                    // words, q8_0 32 = 8 words, w4 128 = 16 words).
                    const auto scale = scales.read(scale_base + wi / group_words);
                    if (w4) {
                        wv[0] = qwen_w4_to_f32(word & 0xFu) * scale;
                        wv[1] = qwen_w4_to_f32((word >> 4u) & 0xFu) * scale;
                        wv[2] = qwen_w4_to_f32((word >> 8u) & 0xFu) * scale;
                        wv[3] = qwen_w4_to_f32((word >> 12u) & 0xFu) * scale;
                        wv[4] = qwen_w4_to_f32((word >> 16u) & 0xFu) * scale;
                        wv[5] = qwen_w4_to_f32((word >> 20u) & 0xFu) * scale;
                        wv[6] = qwen_w4_to_f32((word >> 24u) & 0xFu) * scale;
                        wv[7] = qwen_w4_to_f32((word >> 28u) & 0xFu) * scale;
                    } else if (q8) {
                        wv[0] = qwen_q8_0_to_f32(word & 0xFFu) * scale;
                        wv[1] = qwen_q8_0_to_f32((word >> 8u) & 0xFFu) * scale;
                        wv[2] = qwen_q8_0_to_f32((word >> 16u) & 0xFFu) * scale;
                        wv[3] = qwen_q8_0_to_f32((word >> 24u) & 0xFFu) * scale;
                    } else {
                        wv[0] = lut.read(word & 0xFFu) * scale;
                        wv[1] = lut.read((word >> 8u) & 0xFFu) * scale;
                        wv[2] = lut.read((word >> 16u) & 0xFFu) * scale;
                        wv[3] = lut.read((word >> 24u) & 0xFFu) * scale;
                    }
                } else {
                    wv[0] = qwen_bf16_to_f32(word & 0xFFFFu);
                    wv[1] = qwen_bf16_to_f32(word >> 16u);
                }
                const auto kk = wi * pack;
                for (uint32_t r = 0u; r < mt; r++) {
                    // Ragged m tail: clamp the read to the last valid row (always
                    // in bounds); the epilogue below writes only rows < m.
                    const auto row = min(row0 + r, m - 1u);
                    const auto xb = row * k + kk;
                    for (uint32_t j = 0u; j < pack; j++) {
                        acc[r] = fma(x.read(xb + j), wv[j], acc[r]);
                    }
                }
            };
            // One warp reduction per m row, ALL of them issued before any lane
            // diverges: on HLSL/DX a wave intrinsic that follows a divergent
            // region can execute with the other lanes masked off and return a
            // single lane's partial, which is exactly the class of bug that made
            // m >= 2 wrong on dx while m == 1 (one write) looked fine.
            for (uint32_t r = 0u; r < mt; r++) {
                acc[r] = qwen_warp_butterfly_sum(acc[r]);
            }
            // Then the same epilogue/ABI as the reference GEMM (residual adds
            // y's existing value elementwise).
            $if (lane == 0u) {
                for (uint32_t r = 0u; r < mt; r++) {
                    const auto row = row0 + r;
                    $if (row < m) {
                        const auto idx = row * n + col;
                        Var<float> v = acc[r];
                        if (epilogue == QwenMatmulEpilogue::residual) {
                            v += residual.read(idx);
                        }
                        y.write(idx, v);
                    };
                }
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenGemmMtPlan> plan_qwen35_gemm_mt_dispatch(
    uint64_t m, uint64_t n, uint64_t k, QwenWeightFormat format, uint32_t group,
    const QwenGemmMtConfig &cfg) {
    QwenGemmMtPlan plan;
    plan.mt = cfg.mt;
    plan.columns_per_block = cfg.warps;
    if (auto s = validate_qwen35_gemm_mt_config(cfg); !s.ok()) {
        return Result<QwenGemmMtPlan>::failure(s);
    }
    if (m == 0u || n == 0u || k == 0u) { return Result<QwenGemmMtPlan>::success(plan); }
    // Packed-word contract the kernel relies on (see dsl_ops.h): every weight
    // row must start on a word boundary and one word must not straddle an fp8
    // scale group. Anything else stays on the reference GEMM.
    const uint64_t pack = qwen_weight_pack(format);
    if (k % pack != 0u) {
        return Result<QwenGemmMtPlan>::failure(err_invalid(
            luisa::string{"mt GEMM requires K % "} + luisa::string{std::to_string(pack).c_str()} +
            " == 0 (packed-word aligned weight rows); use make_qwen_gemm_kernel"));
    }
    if (format == QwenWeightFormat::w4) {
        if (auto st = validate_qwen_w4_geometry(k, group); !st.ok()) {
            return Result<QwenGemmMtPlan>::failure(std::move(st));
        }
    } else if (format == QwenWeightFormat::q8_0) {
        if (auto st = validate_qwen_q8_0_geometry(k, group); !st.ok()) {
            return Result<QwenGemmMtPlan>::failure(std::move(st));
        }
    } else if (format == QwenWeightFormat::fp8_e4m3 && (group == 0u || group % pack != 0u)) {
        return Result<QwenGemmMtPlan>::failure(err_invalid(
            "mt GEMM requires an fp8 group that is a nonzero multiple of the packed width (4)"));
    }
    const auto mu = checked_u32(m, "mt GEMM M");
    if (!mu.ok()) { return Result<QwenGemmMtPlan>::failure(mu.status); }
    const auto nu = checked_u32(n, "mt GEMM N");
    if (!nu.ok()) { return Result<QwenGemmMtPlan>::failure(nu.status); }
    const auto ku = checked_u32(k, "mt GEMM K");
    if (!ku.ok()) { return Result<QwenGemmMtPlan>::failure(ku.status); }
    // Element envelopes (the kernels index with uint32): x [M,K], W [N,K],
    // y/residual [M,N], fp8 scales [N, ceil(K/group)].
    const auto xk = checked_product_u32(m, k, "mt GEMM x");
    if (!xk.ok()) { return Result<QwenGemmMtPlan>::failure(xk.status); }
    const auto wn = checked_product_u32(n, k, "mt GEMM weights");
    if (!wn.ok()) { return Result<QwenGemmMtPlan>::failure(wn.status); }
    const auto yn = checked_product_u32(m, n, "mt GEMM output");
    if (!yn.ok()) { return Result<QwenGemmMtPlan>::failure(yn.status); }
    if (format != QwenWeightFormat::bf16) {
        const auto groups_per_row = (k + group - 1u) / group;
        const auto sc = checked_product_u32(n, groups_per_row, "mt GEMM weight scales");
        if (!sc.ok()) { return Result<QwenGemmMtPlan>::failure(sc.status); }
    }
    const uint64_t column_blocks = (n + cfg.warps - 1u) / cfg.warps;
    const uint64_t stripes = (m + cfg.mt - 1u) / cfg.mt;
    // Portable grid envelope: one thread group per column block (x) and one per
    // m stripe (y). Failing closed here keeps the reference GEMM in charge
    // instead of launching a grid the backend rejects (DX refuses > 65535).
    if (column_blocks > kQwenMaxGroupsPerDim) {
        return Result<QwenGemmMtPlan>::failure(err_overflow(
            "mt GEMM column-block count exceeds the 65535-groups-per-dimension envelope "
            "(N is too wide for this kernel family; the reference GEMM accepts it)"));
    }
    if (stripes > kQwenMaxGroupsPerDim) {
        return Result<QwenGemmMtPlan>::failure(err_overflow(
            "mt GEMM m-stripe count exceeds the 65535-groups-per-dimension envelope"));
    }
    const uint64_t block = 32u * cfg.warps;
    const auto threads_x = checked_product_u32(column_blocks, block, "mt GEMM grid x");
    if (!threads_x.ok()) { return Result<QwenGemmMtPlan>::failure(threads_x.status); }
    const auto blocks = checked_product_u32(column_blocks, stripes, "mt GEMM blocks");
    if (!blocks.ok()) { return Result<QwenGemmMtPlan>::failure(blocks.status); }
    plan.total_x = static_cast<uint32_t>(*threads_x.value);
    plan.total_y = static_cast<uint32_t>(stripes);
    plan.column_blocks = static_cast<uint32_t>(column_blocks);
    plan.stripes = static_cast<uint32_t>(stripes);
    plan.blocks = *blocks.value;
    return Result<QwenGemmMtPlan>::success(plan);
}

// ---------------------------------------------------------------------------
// 4c. Activation workspace geometry / aliasing plan (opt_plan/09 V6 + N3.5).
// ---------------------------------------------------------------------------
namespace {

// Per-chunk-row element count of a logical workspace buffer.
enum class WsSel : uint32_t {
    hidden = 0,
    ffn = 1,
    gdn_qkv = 2,
    gdn_v = 3,
    gdn_qk = 4,
    gdn_gate = 5,
    attn_q_proj = 6,
    attn_o_k = 7,
    attn_kv = 8,
};

[[nodiscard]] uint64_t ws_floats_per_row(const QwenDslConfig &cfg, WsSel sel) noexcept {
    switch (sel) {
        case WsSel::hidden: return cfg.hidden;
        case WsSel::ffn: return cfg.ffn;
        case WsSel::gdn_qkv: return cfg.gdn_qkv_n();
        case WsSel::gdn_v: return cfg.gdn_v_n();
        case WsSel::gdn_qk: return cfg.gdn_qk_n();
        case WsSel::gdn_gate: return cfg.gdn_v_heads;
        case WsSel::attn_q_proj: return cfg.attn_q_proj_n();
        case WsSel::attn_o_k: return cfg.attn_o_k();
        case WsSel::attn_kv: return cfg.attn_kv_n();
    }
    return 0u;
}

[[nodiscard]] constexpr uint32_t ws_permanent() noexcept { return ~0u; }

// Arena names. An arena is ONE device buffer; every member below is a handle
// copy of it, so all of them alias byte-identically.
constexpr const char *kWsArenaNames[kQwenWorkspaceArenaCount] = {
    "A0_ffn_gate",
    "A1_ffn_up",
    "A2_ffn_act",
    "A3_z",
    "A4_wy",
    "A5_vnew",
    "A6_q",
    "A7_k",
    "A8_beta",
    "A9_g",
};

struct WsMemberSpec {
    const char *buffer;
    uint32_t arena;// ws_permanent() = never aliased
    QwenWorkspaceStage stage;
    uint32_t first_step;// inclusive command step inside one layer (0 for permanent)
    uint32_t last_step;
    WsSel sel;
};

// The authoritative aliasing plan. Every arena's member list is pairwise
// disjoint in the recorded order documented in dsl_ops.h, EXCEPT that a
// GDN-only member and a full-attention-only member never coexist (a layer is
// exactly one of the two), which is what makes the 2.2x footprint reduction
// possible. `test_qwen_ops_dsl` re-verifies that property from this table, and
// `_gdn_g`/`_gdn_beta` are deliberately kept in their own tiny arenas: the
// decode path fuses the gates step BEFORE the conv, so their live range starts
// at the projection step and overlaps every other GDN buffer.
constexpr WsMemberSpec kWsMembers[] = {
    // permanent (residual stream, live across a whole layer)
    {"_hidden", ws_permanent(), QwenWorkspaceStage::permanent, 0u, 0u, WsSel::hidden},
    {"_normed", ws_permanent(), QwenWorkspaceStage::permanent, 0u, 0u, WsSel::hidden},
    {"_residual", ws_permanent(), QwenWorkspaceStage::permanent, 0u, 0u, WsSel::hidden},
    // A0: FFN gate projection | GDN packed QKV | attention Q projection
    {"_ffn_gate", 0u, QwenWorkspaceStage::ffn, 13u, 15u, WsSel::ffn},
    {"_proj_a", 0u, QwenWorkspaceStage::gdn, 2u, 6u, WsSel::gdn_qkv},
    {"_attn_proj_q", 0u, QwenWorkspaceStage::attention, 2u, 5u, WsSel::attn_q_proj},
    // A1: FFN up projection | GDN conv output | attention Q
    {"_ffn_up", 1u, QwenWorkspaceStage::ffn, 14u, 15u, WsSel::ffn},
    {"_gdn_conv", 1u, QwenWorkspaceStage::gdn, 6u, 9u, WsSel::gdn_qkv},
    {"_attn_q", 1u, QwenWorkspaceStage::attention, 5u, 6u, WsSel::attn_o_k},
    // A2: FFN act (swiglu) | GDN recurrent/chunk output | attention gate
    {"_ffn_act", 2u, QwenWorkspaceStage::ffn, 15u, 16u, WsSel::ffn},
    {"_gdn_out", 2u, QwenWorkspaceStage::gdn, 8u, 10u, WsSel::gdn_v},
    {"_attn_gate", 2u, QwenWorkspaceStage::attention, 5u, 6u, WsSel::attn_o_k},
    // A3: GDN z projection | attention output
    {"_proj_b", 3u, QwenWorkspaceStage::gdn, 3u, 10u, WsSel::gdn_v},
    {"_attn_out", 3u, QwenWorkspaceStage::attention, 6u, 7u, WsSel::attn_o_k},
    // A4: GDN WY w / gated-norm output / alpha | attention K projection
    {"_gdn_w", 4u, QwenWorkspaceStage::gdn, 9u, 9u, WsSel::gdn_v},
    {"_gdn_normed", 4u, QwenWorkspaceStage::gdn, 10u, 11u, WsSel::gdn_v},
    {"_gdn_alpha", 4u, QwenWorkspaceStage::gdn, 4u, 7u, WsSel::gdn_gate},
    {"_attn_proj_k", 4u, QwenWorkspaceStage::attention, 3u, 5u, WsSel::attn_kv},
    // A5: GDN WY v_new / raw beta | attention V projection
    {"_gdn_vnew", 5u, QwenWorkspaceStage::gdn, 9u, 9u, WsSel::gdn_v},
    {"_proj_c", 5u, QwenWorkspaceStage::gdn, 5u, 7u, WsSel::gdn_gate},
    {"_attn_proj_v", 5u, QwenWorkspaceStage::attention, 4u, 5u, WsSel::attn_kv},
    // A6/A7: GDN Q/K | attention K/V
    {"_gdn_q", 6u, QwenWorkspaceStage::gdn, 8u, 9u, WsSel::gdn_qk},
    {"_attn_k", 6u, QwenWorkspaceStage::attention, 5u, 6u, WsSel::attn_kv},
    {"_gdn_k", 7u, QwenWorkspaceStage::gdn, 8u, 9u, WsSel::gdn_qk},
    {"_attn_v", 7u, QwenWorkspaceStage::attention, 5u, 6u, WsSel::attn_kv},
    // A8/A9: GDN decay inputs (live next to every other GDN buffer, see above)
    {"_gdn_beta", 8u, QwenWorkspaceStage::gdn, 4u, 9u, WsSel::gdn_gate},
    {"_gdn_g", 9u, QwenWorkspaceStage::gdn, 4u, 9u, WsSel::gdn_gate},
};

}// namespace

[[nodiscard]] luisa::string_view qwen_workspace_stage_name(QwenWorkspaceStage stage) noexcept {
    switch (stage) {
        case QwenWorkspaceStage::permanent: return "permanent";
        case QwenWorkspaceStage::ffn: return "ffn";
        case QwenWorkspaceStage::gdn: return "gdn";
        case QwenWorkspaceStage::attention: return "attention";
    }
    return "unknown";
}

[[nodiscard]] luisa::vector<QwenWorkspaceMember> qwen_workspace_members(const QwenDslConfig &cfg) {
    luisa::vector<QwenWorkspaceMember> members;
    members.reserve(std::size(kWsMembers));
    for (const auto &spec : kWsMembers) {
        QwenWorkspaceMember m;
        m.buffer = spec.buffer;
        m.arena = spec.arena == ws_permanent() ? "permanent" : kWsArenaNames[spec.arena];
        m.arena_index = spec.arena;
        m.stage = spec.stage;
        m.first_step = spec.first_step;
        m.last_step = spec.last_step;
        m.floats_per_row = ws_floats_per_row(cfg, spec.sel);
        members.push_back(m);
    }
    return members;
}

[[nodiscard]] QwenWorkspaceGeometry qwen_workspace_geometry(const QwenDslConfig &cfg,
                                                            uint64_t rows) noexcept {
    QwenWorkspaceGeometry g;
    g.rows = rows;
    for (const auto &m : qwen_workspace_members(cfg)) {
        if (m.arena_index == ws_permanent()) {
            if (m.buffer == luisa::string_view{"_hidden"}) {
                g.hidden_elems = rows * m.floats_per_row;
            } else if (m.buffer == luisa::string_view{"_normed"}) {
                g.normed_elems = rows * m.floats_per_row;
            } else {
                g.residual_elems = rows * m.floats_per_row;
            }
            continue;
        }
        if (m.arena_index < kQwenWorkspaceArenaCount &&
            m.floats_per_row > g.arena_floats[m.arena_index]) {
            g.arena_floats[m.arena_index] = m.floats_per_row;
        }
    }
    for (uint32_t i = 0u; i < kQwenWorkspaceArenaCount; i++) {
        g.pool_floats_per_row += g.arena_floats[i];
        g.pool_elems += rows * g.arena_floats[i];
    }
    g.chunk_elems = g.pool_elems + g.hidden_elems + g.normed_elems + g.residual_elems;
    g.chunk_bytes = g.chunk_elems * sizeof(float);
    return g;
}

// ---------------------------------------------------------------------------
// 4d. Adaptive outer prefill chunking (opt_plan/02 section 4).
// ---------------------------------------------------------------------------
[[nodiscard]] uint32_t qwen_next_pow2_u32(uint64_t v) noexcept {
    uint64_t p = 1u;
    while (p < v) { p <<= 1u; }
    return p > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t>(p);
}

[[nodiscard]] uint32_t qwen_prefill_chunk_backend_limit(luisa::string_view backend) noexcept {
    // Conservative per-backend ceilings for the adaptive chunk, chosen from the
    // measured Phase 3 runs (see .kimix_cache/qwen_benchmark/phase3/):
    //   * cuda / vk: the plan's 1024 default is reachable (one submit of a
    //     few thousand nodes, no device-lost).
    //   * dx: D3D12 tolerates fewer nodes per submit and the executor's
    //     per-layer sync policy is the only protection, so it stays at half.
    // Unlisted backends get the smallest limit rather than the largest.
    if (backend == luisa::string_view{"cuda"} || backend == luisa::string_view{"vk"}) { return 1024u; }
    return 512u;
}

[[nodiscard]] QwenPrefillChunkPlan qwen_plan_prefill_chunks(uint64_t rows, uint32_t requested_max,
                                                            uint32_t backend_limit) noexcept {
    QwenPrefillChunkPlan plan;
    plan.requested_max = requested_max;
    plan.backend_limit = backend_limit;
    if (requested_max == 0u) { requested_max = 1u; }
    if (backend_limit == 0u) { backend_limit = requested_max; }
    plan.max_chunk = std::min(requested_max, backend_limit);
    plan.limited_by_backend = backend_limit < requested_max;
    // chunk = clamp(next_pow2(rows / 4), 64, max_chunk), then never more than
    // the prompt itself (a short prompt is a single chunk).
    const uint64_t target = rows / 4u;
    uint64_t chunk = qwen_next_pow2_u32(target);
    const uint64_t floor_chunk = std::min<uint64_t>(64u, plan.max_chunk);
    chunk = std::max<uint64_t>(chunk, floor_chunk);
    chunk = std::min<uint64_t>(chunk, plan.max_chunk);
    if (rows != 0u) { chunk = std::min<uint64_t>(chunk, rows); }
    chunk = std::max<uint64_t>(chunk, 1u);
    plan.chunk = static_cast<uint32_t>(chunk);
    plan.chunks = rows == 0u ? 0u : static_cast<uint32_t>((rows + chunk - 1u) / chunk);
    return plan;
}

// ---------------------------------------------------------------------------
// 5. SwiGLU: out[i] = silu(gate[i]) * up[i].
// ---------------------------------------------------------------------------
[[nodiscard]] QwenSwiGluSplitKernel make_qwen_swiglu_split_kernel() noexcept {
    QwenSwiGluSplitKernel kernel = [](BufferFloat gate, BufferFloat up, BufferFloat out,
                                      Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_swiglu_split");
        const auto i = qwen_linear_id();
        $if (i < count) {
            out.write(i, qwen_silu(gate.read(i)) * up.read(i));
        };
    };
    return kernel;
}

[[nodiscard]] QwenSwiGluFusedKernel make_qwen_swiglu_fused_kernel() noexcept {
    QwenSwiGluFusedKernel kernel = [](BufferFloat proj, BufferFloat out,
                                      Var<uint> m, Var<uint> f, Var<uint> padded_cols) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("qwen_swiglu_fused");
        const auto linear = qwen_linear_id();
        const auto row = linear / padded_cols;
        const auto col = linear % padded_cols;
        $if (row < m & col < f) {
            const auto g = proj.read(row * (2u * f) + col);
            const auto u = proj.read(row * (2u * f) + f + col);
            out.write(row * f + col, qwen_silu(g) * u);
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_swiglu_dispatch(uint64_t count) {
    return plan_qwen_threads(count);
}
[[nodiscard]] Result<QwenLaunch2D> plan_qwen_swiglu_fused_dispatch(uint64_t m, uint64_t f) {
    const auto n = checked_product_u32(m, f, "swiglu output");
    if (!n.ok()) { return Result<QwenLaunch2D>::failure(n.status); }
    return plan_qwen_threads_2d(m, f);
}

[[nodiscard]] QwenFfnDecodeKernel make_qwen_ffn_decode_kernel(QwenWeightFormat format) noexcept {
    QwenFfnDecodeKernel kernel = [format](BufferFloat x, BufferUInt gate_w,
                                          BufferFloat gate_scales, BufferUInt up_w,
                                          BufferFloat up_scales, BufferFloat out,
                                          Var<uint> f, Var<uint> k, Var<uint> group) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(luisa::string{"qwen_ffn_decode"} + qwen_format_name_suffix(format));
        Shared<float> lut{format == QwenWeightFormat::fp8_e4m3 ? 256u : 1u};
        if (format == QwenWeightFormat::fp8_e4m3) {
            $for (i, thread_x(), 256u, 128u) { lut[i] = qwen_fp8_e4m3_to_f32(i); };
            sync_block();
        }
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < f & k > 0u) {
            const auto words = format == QwenWeightFormat::bf16 ? (k >> 1u)
                             : format == QwenWeightFormat::w4  ? (k >> 3u)
                                                               : (k >> 2u);
            const auto element_base = warp * k;
            const auto word_base = format == QwenWeightFormat::bf16
                                       ? element_base / 2u
                                   : format == QwenWeightFormat::w4
                                       ? element_base / 8u
                                       : element_base / 4u;
            Var<uint> scale_base = def(0u);
            if (format != QwenWeightFormat::bf16) {
                scale_base = warp * ((k + group - 1u) / group);
            }
            const auto group_words = format == QwenWeightFormat::w4 ? (group >> 3u) : (group >> 2u);
            Var<float> gate_acc = def(0.0f), up_acc = def(0.0f);
            $for (wi, lane, words, 32u) {
                const auto gw = gate_w.read(word_base + wi);
                const auto uw = up_w.read(word_base + wi);
                if (format == QwenWeightFormat::w4) {
                    // One word = 8 int4 codes per stream; one group scale per
                    // word (8 products hoisted per scale load).
                    const auto ki = wi * 8u;
                    const auto gs = gate_scales.read(scale_base + wi / group_words);
                    const auto us = up_scales.read(scale_base + wi / group_words);
                    const auto x0 = x.read(ki);
                    const auto x1 = x.read(ki + 1u);
                    const auto x2 = x.read(ki + 2u);
                    const auto x3 = x.read(ki + 3u);
                    const auto x4 = x.read(ki + 4u);
                    const auto x5 = x.read(ki + 5u);
                    const auto x6 = x.read(ki + 6u);
                    const auto x7 = x.read(ki + 7u);
                    gate_acc += x0 * qwen_w4_to_f32(gw & 0xFu) * gs;
                    gate_acc += x1 * qwen_w4_to_f32((gw >> 4u) & 0xFu) * gs;
                    gate_acc += x2 * qwen_w4_to_f32((gw >> 8u) & 0xFu) * gs;
                    gate_acc += x3 * qwen_w4_to_f32((gw >> 12u) & 0xFu) * gs;
                    gate_acc += x4 * qwen_w4_to_f32((gw >> 16u) & 0xFu) * gs;
                    gate_acc += x5 * qwen_w4_to_f32((gw >> 20u) & 0xFu) * gs;
                    gate_acc += x6 * qwen_w4_to_f32((gw >> 24u) & 0xFu) * gs;
                    gate_acc += x7 * qwen_w4_to_f32((gw >> 28u) & 0xFu) * gs;
                    up_acc += x0 * qwen_w4_to_f32(uw & 0xFu) * us;
                    up_acc += x1 * qwen_w4_to_f32((uw >> 4u) & 0xFu) * us;
                    up_acc += x2 * qwen_w4_to_f32((uw >> 8u) & 0xFu) * us;
                    up_acc += x3 * qwen_w4_to_f32((uw >> 12u) & 0xFu) * us;
                    up_acc += x4 * qwen_w4_to_f32((uw >> 16u) & 0xFu) * us;
                    up_acc += x5 * qwen_w4_to_f32((uw >> 20u) & 0xFu) * us;
                    up_acc += x6 * qwen_w4_to_f32((uw >> 24u) & 0xFu) * us;
                    up_acc += x7 * qwen_w4_to_f32((uw >> 28u) & 0xFu) * us;
                } else if (format == QwenWeightFormat::bf16) {
                    const auto ki = wi * 2u;
                    const auto x0 = x.read(ki);
                    const auto x1 = x.read(ki + 1u);
                    gate_acc += x0 * qwen_bf16_to_f32(gw & 0xFFFFu);
                    gate_acc += x1 * qwen_bf16_to_f32(gw >> 16u);
                    up_acc += x0 * qwen_bf16_to_f32(uw & 0xFFFFu);
                    up_acc += x1 * qwen_bf16_to_f32(uw >> 16u);
                } else {
                    const auto ki = wi * 4u;
                    const auto gs = gate_scales.read(scale_base + wi / group_words);
                    const auto us = up_scales.read(scale_base + wi / group_words);
                    const auto x0 = x.read(ki);
                    const auto x1 = x.read(ki + 1u);
                    const auto x2 = x.read(ki + 2u);
                    const auto x3 = x.read(ki + 3u);
                    if (format == QwenWeightFormat::q8_0) {
                        // Exact signed-int8 decode (no table, no barrier).
                        gate_acc += x0 * qwen_q8_0_to_f32(gw & 0xFFu) * gs;
                        gate_acc += x1 * qwen_q8_0_to_f32((gw >> 8u) & 0xFFu) * gs;
                        gate_acc += x2 * qwen_q8_0_to_f32((gw >> 16u) & 0xFFu) * gs;
                        gate_acc += x3 * qwen_q8_0_to_f32((gw >> 24u) & 0xFFu) * gs;
                        up_acc += x0 * qwen_q8_0_to_f32(uw & 0xFFu) * us;
                        up_acc += x1 * qwen_q8_0_to_f32((uw >> 8u) & 0xFFu) * us;
                        up_acc += x2 * qwen_q8_0_to_f32((uw >> 16u) & 0xFFu) * us;
                        up_acc += x3 * qwen_q8_0_to_f32((uw >> 24u) & 0xFFu) * us;
                    } else {
                        gate_acc += x0 * lut.read(gw & 0xFFu) * gs;
                        gate_acc += x1 * lut.read((gw >> 8u) & 0xFFu) * gs;
                        gate_acc += x2 * lut.read((gw >> 16u) & 0xFFu) * gs;
                        gate_acc += x3 * lut.read((gw >> 24u) & 0xFFu) * gs;
                        up_acc += x0 * lut.read(uw & 0xFFu) * us;
                        up_acc += x1 * lut.read((uw >> 8u) & 0xFFu) * us;
                        up_acc += x2 * lut.read((uw >> 16u) & 0xFFu) * us;
                        up_acc += x3 * lut.read((uw >> 24u) & 0xFFu) * us;
                    }
                }
            };
            gate_acc = warp_active_sum(gate_acc);
            up_acc = warp_active_sum(up_acc);
            $if (lane == 0u) { out.write(warp, qwen_silu(gate_acc) * up_acc); };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_ffn_decode_dispatch(
    uint64_t f, uint64_t k, QwenWeightFormat format, uint32_t group) {
    const uint64_t pack = qwen_weight_pack(format);
    if (k == 0u || k % pack != 0u ||
        (format == QwenWeightFormat::fp8_e4m3 &&
         (group == 0u || group % 4u != 0u))) {
        return Result<QwenLaunch1D>::failure(err_invalid("fused FFN decode K/group is not packed-word aligned"));
    }
    if (format == QwenWeightFormat::w4) {
        if (auto st = validate_qwen_w4_geometry(k, group); !st.ok()) {
            return Result<QwenLaunch1D>::failure(std::move(st));
        }
    }
    if (format == QwenWeightFormat::q8_0) {
        if (auto st = validate_qwen_q8_0_geometry(k, group); !st.ok()) {
            return Result<QwenLaunch1D>::failure(std::move(st));
        }
    }
    const auto xk = checked_product_u32(1u, k, "fused FFN x");
    const auto wn = checked_product_u32(f, k, "fused FFN weights");
    if (!xk.ok()) { return Result<QwenLaunch1D>::failure(xk.status); }
    if (!wn.ok()) { return Result<QwenLaunch1D>::failure(wn.status); }
    return plan_warp_rows(f, 128u);
}

// 128-bit twin of the fused FFN decode kernel (opt_plan/09 N2). Two packed words
// per accumulator per stream: 128-bit weight loads, one FP8 scale per uint4,
// float4 activations, branchless LUT-free E4M3 decode. Same SiLU(gate)*up
// epilogue and the same two warp reductions.
[[nodiscard]] QwenFfnDecodePackedKernel make_qwen_ffn_decode_packed_kernel(
    QwenWeightFormat format) noexcept {
    const bool fp8 = format == QwenWeightFormat::fp8_e4m3;
    const bool q8 = format == QwenWeightFormat::q8_0;
    const bool scaled = fp8 || q8;
    QwenFfnDecodePackedKernel kernel = [fp8, q8, scaled, format](BufferFloat4 x, BufferUInt4 gate_w,
                                             BufferFloat gate_scales, BufferUInt4 up_w,
                                             BufferFloat up_scales, BufferFloat out,
                                             Var<uint> f, Var<uint> k, Var<uint> group,
                                             Var<uint> vecs) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(luisa::string{"qwen_ffn_decode_packed"} + qwen_format_name_suffix(format));
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < f & k > 0u) {
            const auto w_base = warp * vecs;
            Var<uint> scale_base = def(0u);
            if (scaled) { scale_base = warp * ((k + group - 1u) / group); }
            Var<uint> group_vecs = def(1u);
            if (scaled) { group_vecs = group >> 4u; }
            // Two accumulators per stream keep two independent FMA chains in
            // flight per weight stream without an extra live register set.
            Var<float> g0 = def(0.0f), g1 = def(0.0f), u0 = def(0.0f), u1 = def(0.0f);
            $for (vi, lane, vecs, 32u) {
                const auto gw = gate_w.read(w_base + vi);
                const auto uw = up_w.read(w_base + vi);
                if (scaled) {
                    const auto s_gate = gate_scales.read(scale_base + vi / group_vecs);
                    const auto s_up = up_scales.read(scale_base + vi / group_vecs);
                    const auto v0 = x.read(vi * 4u);
                    const auto v1 = x.read(vi * 4u + 1u);
                    const auto v2 = x.read(vi * 4u + 2u);
                    const auto v3 = x.read(vi * 4u + 3u);
                    g0 += v0.x * qwen_scaled_byte_to_f32(gw.x & 0xFFu, format) * s_gate;
                    g0 += v0.y * qwen_scaled_byte_to_f32((gw.x >> 8u) & 0xFFu, format) * s_gate;
                    g0 += v0.z * qwen_scaled_byte_to_f32((gw.x >> 16u) & 0xFFu, format) * s_gate;
                    g0 += v0.w * qwen_scaled_byte_to_f32((gw.x >> 24u) & 0xFFu, format) * s_gate;
                    g1 += v1.x * qwen_scaled_byte_to_f32(gw.y & 0xFFu, format) * s_gate;
                    g1 += v1.y * qwen_scaled_byte_to_f32((gw.y >> 8u) & 0xFFu, format) * s_gate;
                    g1 += v1.z * qwen_scaled_byte_to_f32((gw.y >> 16u) & 0xFFu, format) * s_gate;
                    g1 += v1.w * qwen_scaled_byte_to_f32((gw.y >> 24u) & 0xFFu, format) * s_gate;
                    u0 += v0.x * qwen_scaled_byte_to_f32(uw.x & 0xFFu, format) * s_up;
                    u0 += v0.y * qwen_scaled_byte_to_f32((uw.x >> 8u) & 0xFFu, format) * s_up;
                    u0 += v0.z * qwen_scaled_byte_to_f32((uw.x >> 16u) & 0xFFu, format) * s_up;
                    u0 += v0.w * qwen_scaled_byte_to_f32((uw.x >> 24u) & 0xFFu, format) * s_up;
                    u1 += v1.x * qwen_scaled_byte_to_f32(uw.y & 0xFFu, format) * s_up;
                    u1 += v1.y * qwen_scaled_byte_to_f32((uw.y >> 8u) & 0xFFu, format) * s_up;
                    u1 += v1.z * qwen_scaled_byte_to_f32((uw.y >> 16u) & 0xFFu, format) * s_up;
                    u1 += v1.w * qwen_scaled_byte_to_f32((uw.y >> 24u) & 0xFFu, format) * s_up;
                    g0 += v2.x * qwen_scaled_byte_to_f32(gw.z & 0xFFu, format) * s_gate;
                    g0 += v2.y * qwen_scaled_byte_to_f32((gw.z >> 8u) & 0xFFu, format) * s_gate;
                    g0 += v2.z * qwen_scaled_byte_to_f32((gw.z >> 16u) & 0xFFu, format) * s_gate;
                    g0 += v2.w * qwen_scaled_byte_to_f32((gw.z >> 24u) & 0xFFu, format) * s_gate;
                    g1 += v3.x * qwen_scaled_byte_to_f32(gw.w & 0xFFu, format) * s_gate;
                    g1 += v3.y * qwen_scaled_byte_to_f32((gw.w >> 8u) & 0xFFu, format) * s_gate;
                    g1 += v3.z * qwen_scaled_byte_to_f32((gw.w >> 16u) & 0xFFu, format) * s_gate;
                    g1 += v3.w * qwen_scaled_byte_to_f32((gw.w >> 24u) & 0xFFu, format) * s_gate;
                    u0 += v2.x * qwen_scaled_byte_to_f32(uw.z & 0xFFu, format) * s_up;
                    u0 += v2.y * qwen_scaled_byte_to_f32((uw.z >> 8u) & 0xFFu, format) * s_up;
                    u0 += v2.z * qwen_scaled_byte_to_f32((uw.z >> 16u) & 0xFFu, format) * s_up;
                    u0 += v2.w * qwen_scaled_byte_to_f32((uw.z >> 24u) & 0xFFu, format) * s_up;
                    u1 += v3.x * qwen_scaled_byte_to_f32(uw.w & 0xFFu, format) * s_up;
                    u1 += v3.y * qwen_scaled_byte_to_f32((uw.w >> 8u) & 0xFFu, format) * s_up;
                    u1 += v3.z * qwen_scaled_byte_to_f32((uw.w >> 16u) & 0xFFu, format) * s_up;
                    u1 += v3.w * qwen_scaled_byte_to_f32((uw.w >> 24u) & 0xFFu, format) * s_up;
                } else {
                    const auto v0 = x.read(vi * 2u);
                    const auto v1 = x.read(vi * 2u + 1u);
                    g0 += v0.x * qwen_bf16_to_f32(gw.x & 0xFFFFu);
                    g0 += v0.y * qwen_bf16_to_f32((gw.x >> 16u) & 0xFFFFu);
                    g1 += v0.z * qwen_bf16_to_f32(gw.y & 0xFFFFu);
                    g1 += v0.w * qwen_bf16_to_f32((gw.y >> 16u) & 0xFFFFu);
                    u0 += v0.x * qwen_bf16_to_f32(uw.x & 0xFFFFu);
                    u0 += v0.y * qwen_bf16_to_f32((uw.x >> 16u) & 0xFFFFu);
                    u1 += v0.z * qwen_bf16_to_f32(uw.y & 0xFFFFu);
                    u1 += v0.w * qwen_bf16_to_f32((uw.y >> 16u) & 0xFFFFu);
                    g0 += v1.x * qwen_bf16_to_f32(gw.z & 0xFFFFu);
                    g0 += v1.y * qwen_bf16_to_f32((gw.z >> 16u) & 0xFFFFu);
                    g1 += v1.z * qwen_bf16_to_f32(gw.w & 0xFFFFu);
                    g1 += v1.w * qwen_bf16_to_f32((gw.w >> 16u) & 0xFFFFu);
                    u0 += v1.x * qwen_bf16_to_f32(uw.z & 0xFFFFu);
                    u0 += v1.y * qwen_bf16_to_f32((uw.z >> 16u) & 0xFFFFu);
                    u1 += v1.z * qwen_bf16_to_f32(uw.w & 0xFFFFu);
                    u1 += v1.w * qwen_bf16_to_f32((uw.w >> 16u) & 0xFFFFu);
                }
            };
            const auto gate_acc = warp_active_sum(g0 + g1);
            const auto up_acc = warp_active_sum(u0 + u1);
            $if (lane == 0u) { out.write(warp, qwen_silu(gate_acc) * up_acc); };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_ffn_decode_packed_dispatch(
    uint64_t f, uint64_t k, QwenWeightFormat format, uint32_t group) {
    if (auto s = validate_packed_decode_geometry(k, format, group); !s.ok()) {
        return Result<QwenLaunch1D>::failure(std::move(s));
    }
    const auto wn = checked_product_u32(f, k, "fused FFN packed weights");
    if (!wn.ok()) { return Result<QwenLaunch1D>::failure(wn.status); }
    return plan_warp_rows(f, 128u);
}

// ---------------------------------------------------------------------------
// 6. Elementwise utilities.
// ---------------------------------------------------------------------------
[[nodiscard]] QwenResidualAddKernel make_qwen_residual_add_kernel() noexcept {
    QwenResidualAddKernel kernel = [](BufferFloat a, BufferFloat b, BufferFloat out,
                                      Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_residual_add");
        const auto i = qwen_linear_id();
        $if (i < count) { out.write(i, a.read(i) + b.read(i)); };
    };
    return kernel;
}

[[nodiscard]] QwenZeroFillKernel make_qwen_zero_fill_kernel() noexcept {
    QwenZeroFillKernel kernel = [](BufferFloat out, Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_zero_fill");
        const auto i = qwen_linear_id();
        $if (i < count) { out.write(i, 0.0f); };
    };
    return kernel;
}

[[nodiscard]] QwenCopyKernel make_qwen_copy_kernel() noexcept {
    QwenCopyKernel kernel = [](BufferFloat src, BufferFloat out, Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_copy");
  const auto i = qwen_linear_id();
    $if (i < count) { out.write(i, src.read(i)); };
  };
  return kernel;
}

[[nodiscard]] QwenCopyUKernel make_qwen_copy_u_kernel() noexcept {
  QwenCopyUKernel kernel = [](BufferUInt src, BufferUInt out, Var<uint> count) noexcept {
    set_block_size(256u, 1u, 1u);
    set_name("qwen_copy_u32");
    const auto i = qwen_linear_id();
    $if (i < count) { out.write(i, src.read(i)); };
  };
  return kernel;
}

[[nodiscard]] QwenImportanceAccumulateKernel make_qwen_importance_accumulate_kernel() noexcept {
    QwenImportanceAccumulateKernel kernel = [](BufferFloat x, BufferFloat acc, Var<uint> rows,
                                               Var<uint> k) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_importance_accumulate");
        const auto j = qwen_linear_id();
        $if (j < k) {
            Var<float> sum = acc.read(j);
            $for (r, 0u, rows, 1u) {
                const auto v = x.read(r * k + j);
                sum += v * v;
            };
            acc.write(j, sum);
        };
    };
    return kernel;
}

[[nodiscard]] QwenCopyIKernel make_qwen_copy_i_kernel() noexcept {
  QwenCopyIKernel kernel = [](BufferInt src, BufferInt out, Var<uint> count) noexcept {
    set_block_size(256u, 1u, 1u);
    set_name("qwen_copy_i32");
    const auto i = qwen_linear_id();
    $if (i < count) { out.write(i, src.read(i)); };
  };
  return kernel;
}

[[nodiscard]] QwenGreedyArgmaxKernel make_qwen_greedy_argmax_kernel() noexcept {
    QwenGreedyArgmaxKernel kernel = [](BufferFloat logits, BufferUInt next_ids,
                                       BufferUInt history, Var<uint> vocab,
                                       Var<uint> slot) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_greedy_argmax");
        Shared<float> values{256u};
        Shared<uint> indices{256u};
        Shared<uint> invalid{256u};
        const auto tid = thread_x();
        Var<float> best = as<float>(0xFF800000u); // negative infinity, including all -inf rows
        Var<uint> best_index = def(vocab);
        Var<uint> bad = def(0u);
        $for (i, tid, vocab, 256u) {
            const auto v = logits.read(i);
            $if (isnan(v)) {
                bad = 1u;
            } $else {
                $if (v > best | (v == best & i < best_index)) {
                    best = v;
                    best_index = i;
                };
            };
        };
        values[tid] = best;
        indices[tid] = best_index;
        invalid[tid] = bad;
        sync_block();
        // Each step reads the upper half and writes only the disjoint lower
        // half, so one barrier per level is sufficient. Explicit index compare
        // makes the result independent of reduction-tree shape.
        for (uint32_t stride = 128u; stride != 0u; stride >>= 1u) {
            $if (tid < stride) {
                const auto other_v = values.read(tid + stride);
                const auto other_i = indices.read(tid + stride);
                $if (other_v > values.read(tid) |
                     (other_v == values.read(tid) & other_i < indices.read(tid))) {
                    values[tid] = other_v;
                    indices[tid] = other_i;
                };
                invalid[tid] = invalid.read(tid) | invalid.read(tid + stride);
            };
            sync_block();
        }
        $if (tid == 0u) {
            Var<uint> token = indices.read(0u);
            $if (invalid.read(0u) != 0u) { token = vocab; };
            next_ids.write(0u, token);
            history.write(slot, token);
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_greedy_argmax_dispatch(uint64_t vocab) {
    if (vocab == 0u || vocab > 0xFFFFFFFFull) {
        return Result<QwenLaunch1D>::failure(err_invalid("greedy argmax vocab must be in [1,uint32_max]"));
    }
    auto plan = plan_qwen_threads(256u, 256u);
    if (plan.ok()) { plan.value->work_items = 1u; }
    return plan;
}

namespace {

void qwen_greedy_reduce_shared(Shared<float> &values, Shared<uint> &indices,
                               Shared<uint> &invalid) noexcept {
    sync_block();
    for (uint32_t stride = 128u; stride != 0u; stride >>= 1u) {
        const auto tid = thread_x();
        $if (tid < stride) {
            const auto other_v = values.read(tid + stride);
            const auto other_i = indices.read(tid + stride);
            $if (other_v > values.read(tid) |
                 (other_v == values.read(tid) & other_i < indices.read(tid))) {
                values[tid] = other_v;
                indices[tid] = other_i;
            };
            invalid[tid] = invalid.read(tid) | invalid.read(tid + stride);
        };
        sync_block();
    }
}

}// namespace

[[nodiscard]] QwenGreedyArgmaxPartialKernel make_qwen_greedy_argmax_partial_kernel() noexcept {
    QwenGreedyArgmaxPartialKernel kernel = [](BufferFloat logits, BufferFloat out_values,
                                              BufferUInt out_indices, BufferUInt out_invalid,
                                              Var<uint> vocab) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_greedy_argmax_partial");
        Shared<float> values{256u};
        Shared<uint> indices{256u};
        Shared<uint> invalid{256u};
        const auto block = qwen_linear_id() / 256u;
        const auto tid = thread_x();
        Var<float> best = as<float>(0xFF800000u);
        Var<uint> best_index = def(vocab);
        Var<uint> bad = def(0u);
        $for (i, block * 256u + tid, vocab, qwen_greedy_argmax_blocks * 256u) {
            const auto v = logits.read(i);
            $if (isnan(v)) {
                bad = 1u;
            } $else {
                $if (v > best | (v == best & i < best_index)) {
                    best = v;
                    best_index = i;
                };
            };
        };
        values[tid] = best;
        indices[tid] = best_index;
        invalid[tid] = bad;
        qwen_greedy_reduce_shared(values, indices, invalid);
        $if (tid == 0u) {
            out_values.write(block, values.read(0u));
            out_indices.write(block, indices.read(0u));
            out_invalid.write(block, invalid.read(0u));
        };
    };
    return kernel;
}

[[nodiscard]] QwenGreedyArgmaxFinishKernel make_qwen_greedy_argmax_finish_kernel() noexcept {
    QwenGreedyArgmaxFinishKernel kernel = [](BufferFloat in_values, BufferUInt in_indices,
                                             BufferUInt in_invalid, BufferUInt next_ids,
                                             BufferUInt history, Var<uint> vocab,
                                             Var<uint> slot) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_greedy_argmax_finish");
        Shared<float> values{256u};
        Shared<uint> indices{256u};
        Shared<uint> invalid{256u};
        const auto tid = thread_x();
        values[tid] = in_values.read(tid);
        indices[tid] = in_indices.read(tid);
        invalid[tid] = in_invalid.read(tid);
        qwen_greedy_reduce_shared(values, indices, invalid);
        $if (tid == 0u) {
            Var<uint> token = indices.read(0u);
            $if (invalid.read(0u) != 0u) { token = vocab; };
            next_ids.write(0u, token);
            history.write(slot, token);
        };
    };
    return kernel;
}

[[nodiscard]] QwenGreedyArgmaxFinishDevKernel make_qwen_greedy_argmax_finish_dev_kernel() noexcept {
    QwenGreedyArgmaxFinishDevKernel kernel = [](BufferFloat in_values, BufferUInt in_indices,
                                                BufferUInt in_invalid, BufferUInt next_ids,
                                                BufferUInt history, BufferUInt state,
                                                Var<uint> vocab, Var<uint> history_capacity) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_greedy_argmax_finish_dev");
        Shared<float> values{256u};
        Shared<uint> indices{256u};
        Shared<uint> invalid{256u};
        const auto tid = thread_x();
        values[tid] = in_values.read(tid);
        indices[tid] = in_indices.read(tid);
        invalid[tid] = in_invalid.read(tid);
        qwen_greedy_reduce_shared(values, indices, invalid);
        // ONE thread owns every store, so the state update is a single ordered
        // sequence inside the graph: read the slot and position, store the
        // token, then advance both. kv_len is re-derived from the advanced
        // position (kv_len == position + 1), never accumulated independently.
        $if (tid == 0u) {
            Var<uint> token = indices.read(0u);
            $if (invalid.read(0u) != 0u) { token = vocab; };
            next_ids.write(0u, token);
            const auto slot = state.read(qwen_dec_state_history_slot);
            $if (slot < history_capacity) { history.write(slot, token); };
            const auto next_position = state.read(qwen_dec_state_position) + 1u;
            state.write(qwen_dec_state_position, next_position);
            state.write(qwen_dec_state_kv_len, next_position + 1u);
            state.write(qwen_dec_state_history_slot, slot + 1u);
        };
    };
    return kernel;
}

[[nodiscard]] QwenGreedyArgmaxFinishWhileKernel make_qwen_greedy_argmax_finish_while_kernel() noexcept {
    QwenGreedyArgmaxFinishWhileKernel kernel = [](BufferFloat in_values, BufferUInt in_indices,
                                                  BufferUInt in_invalid, BufferUInt next_ids,
                                                  BufferUInt history, BufferUInt state,
                                                  Var<uint> vocab, Var<uint> history_capacity,
                                                  Var<uint64_t> cond_handle,
                                                  Var<uint> eos_token) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_greedy_argmax_finish_while");
        Shared<float> values{256u};
        Shared<uint> indices{256u};
        Shared<uint> invalid{256u};
        const auto tid = thread_x();
        values[tid] = in_values.read(tid);
        indices[tid] = in_indices.read(tid);
        invalid[tid] = in_invalid.read(tid);
        qwen_greedy_reduce_shared(values, indices, invalid);
        // Selection, history store and the state advance are IDENTICAL to
        // make_qwen_greedy_argmax_finish_dev_kernel, so a chunked run produces the
        // same ids, the same history slots and the same position as a per-token run.
        // The extra work is the conditional-WHILE bookkeeping, and it runs in the same
        // single owning thread, so the condition is written strictly after this
        // iteration's stores:
        //  * [remaining] is the launch budget, decremented once per executed step;
        //  * [eos] records whether THIS step selected EOS (or an invalid token);
        //  * the condition is re-armed with the device-computed continuation flag.
        // The loop therefore stops on the device -- on EOS, on an invalid selection, at
        // the context limit or when the budget is spent -- instead of running a fixed
        // trip count and leaving the host to guess what ran.
        $if (tid == 0u) {
            Var<uint> token = indices.read(0u);
            const auto invalid_selection = invalid.read(0u) != 0u;
            $if (invalid_selection) { token = vocab; };
            next_ids.write(0u, token);
            const auto slot = state.read(qwen_dec_state_history_slot);
            $if (slot < history_capacity) { history.write(slot, token); };
            const auto next_position = state.read(qwen_dec_state_position) + 1u;
            state.write(qwen_dec_state_position, next_position);
            state.write(qwen_dec_state_kv_len, next_position + 1u);
            state.write(qwen_dec_state_history_slot, slot + 1u);
            // Budget: saturating decrement, so an unset budget can never wrap.
            const auto remaining_before = state.read(qwen_dec_state_remaining);
            Var<uint> remaining = 0u;
            $if (remaining_before != 0u) { remaining = remaining_before - 1u; };
            state.write(qwen_dec_state_remaining, remaining);
            // [eos] states WHY the selection itself ends generation: EOS, the NaN
            // sentinel or any id the host would reject (`token >= vocab`). It is kept
            // separate from the loop's continuation flag so the host can tell an EOS
            // stop from a context-limit stop after the launch.
            Var<uint> eos_flag = 0u;
            $if (invalid_selection | (token >= vocab) | (token == eos_token)) {
                eos_flag = 1u;
            };
            state.write(qwen_dec_state_eos, eos_flag);
            // Continuation: budget left AND the selection did not end generation AND the
            // context can still accept another token.
            Var<uint> cont = 0u;
            $if (remaining != 0u) {
                $if (eos_flag == 0u) {
                    $if (next_position < history_capacity) { cont = 1u; };
                };
            };
            set_graph_conditional(cond_handle, cont);
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_greedy_argmax_partial_dispatch(uint64_t vocab) {
    if (vocab == 0u || vocab > 0xFFFFFFFFull) {
        return Result<QwenLaunch1D>::failure(err_invalid("greedy partial argmax vocab is invalid"));
    }
    auto plan = plan_qwen_threads(qwen_greedy_argmax_blocks * 256u, 256u);
    if (plan.ok()) { plan.value->work_items = qwen_greedy_argmax_blocks; }
    return plan;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_greedy_argmax_finish_dispatch(uint64_t vocab) {
    return plan_qwen_greedy_argmax_dispatch(vocab);
}

// ---------------------------------------------------------------------------
// 7. Full attention
// ---------------------------------------------------------------------------
namespace {

// NeoX rotate-half partial RoPE for one lane-owned pair (i, i+half) of a
// head row, half = rope_dim/2. inv_freq[i] = theta^(-2i/rope_dim).
// Text mRoPE: the three position axes collapse to the same scalar position,
// so one frequency table serves all of them (HF recomposition_frequencies
// duplicates cos/sin across the two halves).
[[nodiscard]] auto qwen_rope_pair(Var<float> x0, Var<float> x1, Var<float> pos,
                                  Var<uint> pair, Var<uint> rope_dim, Var<float> theta) noexcept {
    const auto angle = pos * exp2(log2(theta) * (-cast<float>(2u * pair) / cast<float>(rope_dim)));
    const auto c = cos(angle);
    const auto s = sin(angle);
    return compose(x0 * c - x1 * s, x0 * s + x1 * c);
}

}// namespace

namespace {
// ---------------------------------------------------------------------------
// KV cache addressing (the device mirror of qwen/kv_cache_layout.h).
//
// Every attention kernel reaches the K/V cache through these three helpers, so the
// dtype (one FP32 code per word vs two BF16 codes per word) and the layout
// (position-major legacy vs 64-position blocked, head-major) are decided in ONE
// place. `kv` is a host-side compile-time constant of the kernel (QwenKvStorage),
// so `if (!blocked)` and `if (!packed)` fold away entirely.
//
// A BF16 word must never be read as a float VALUE: qwen_kv_load takes the word,
// bit-casts it to uint32 and widens the selected 16-bit half with
// qwen_bf16_to_f32, which is exact. Activations, accumulators and the softmax stay
// FP32 on every path.
// ---------------------------------------------------------------------------
// Code index of (position, kv_head, channel) inside one K or V region. Identical
// arithmetic to QwenKvLayoutDesc::code_index: the region base is owned by the
// allocator's BufferView, so only this offset is computed in-kernel.
[[nodiscard]] inline auto qwen_kv_code_index(Var<uint> pos, Var<uint> head, Var<uint> channel,
                                             Var<uint> kv_heads, Var<uint> head_dim,
                                             Var<uint> cache_capacity, bool blocked) noexcept {
    if (!blocked) {
        return Var<uint>{pos * (kv_heads * head_dim) + head * head_dim + channel};
    }
    const auto blocks = (cache_capacity + (qwen_kv_position_block - 1u)) / qwen_kv_position_block;
    const auto head_stride = blocks * (qwen_kv_position_block * head_dim);
    return Var<uint>{head * head_stride +
                     (pos / qwen_kv_position_block) * (qwen_kv_position_block * head_dim) +
                     (pos % qwen_kv_position_block) * head_dim + channel};
}
    // Stored value of one code with the dtype's own codec (no scale applied yet).
    // fp8_e4m3 decodes with the SAME branchless E4M3 table-free helper the weight
    // GEMV uses, so a KV code and a weight code can never decode differently.
    [[nodiscard]] inline auto qwen_kv_code_value(const BufferFloat &cache, Var<uint> code,
                                                QwenKvStorage kv) noexcept {
        if (!kv.packed_words()) { return Var<float>{cache.read(code)}; }
        const auto word = as<uint>(cache.read(code / kv.elements_per_word()));
        const auto shift = (code % kv.elements_per_word()) * (8u * qwen_kv_element_bytes(kv.dtype));
        if (kv.dtype == QwenKvDType::bf16) {
            return Var<float>{qwen_bf16_to_f32((word >> shift) & 0xFFFFu)};
        }
        return Var<float>{qwen_fp8_e4m3_to_f32_branchless((word >> shift) & 0xFFu)};
    }
    // The FP32 scale of the (kv_head, 64-position block) a position belongs to, or
    // 1 for an unscaled dtype (the branch folds away and the dummy buffer is never
    // read). Every read of an fp8 code MUST go through this: an fp8 value is code *
    // scale and a read with the wrong block scale is exactly the bug the online
    // append design exists to prevent.
    [[nodiscard]] inline auto qwen_kv_scale_at(const BufferFloat &scale, Var<uint> head, Var<uint> pos,
                                               Var<uint> cache_capacity, QwenKvStorage kv) noexcept {
        if (!kv.scaled()) { return Var<float>{1.0f}; }
        const auto blocks = (cache_capacity + (qwen_kv_position_block - 1u)) / qwen_kv_position_block;
        return Var<float>{scale.read(head * blocks + pos / qwen_kv_position_block)};
    }
    // Index + decode + scale in one step: the only way the attention kernels read a
    // cached K/V value.
    [[nodiscard]] inline auto qwen_kv_load_at(const BufferFloat &cache, const BufferFloat &scale,
                                              Var<uint> pos, Var<uint> head, Var<uint> channel,
                                              Var<uint> kv_heads, Var<uint> head_dim,
                                              Var<uint> cache_capacity, QwenKvStorage kv) noexcept {
        const auto code = qwen_kv_code_index(pos, head, channel, kv_heads, head_dim,
                                             cache_capacity, kv.blocked());
        return Var<float>{qwen_kv_code_value(cache, code, kv) *
                      qwen_kv_scale_at(scale, head, pos, cache_capacity, kv)};
}

// RNE FP32 -> BF16 code, bit-identical to qwen::f32_to_bf16 (ties to even, carries
// into the exponent, NaN -> quiet NaN, overflow -> +/-inf).
[[nodiscard]] inline auto qwen_bf16_from_f32(Var<float> v) noexcept {
    const auto bits = as<uint>(v);
    const auto is_nan = ((bits & 0x7F800000u) == 0x7F800000u) & ((bits & 0x7FFFFFu) != 0u);
    const auto rounded = (bits + (0x7FFFu + ((bits >> 16u) & 1u))) >> 16u;
    return Var<uint>{ite(is_nan, (bits >> 16u) | 0x0040u, rounded)};
}

// Round a NON-NEGATIVE float to the nearest INTEGER-valued float with ties to
// even. Every caller keeps the argument below 2^24 (E4M3 has at most 15
// significand units), so the result is an exactly representable integer and the
// conversion is exact. This is the device half of the E4M3 RNE rule
// qwen::f32_to_fp8_e4m3 implements in integer arithmetic; the codec test compares
// the two bit for bit over the whole code space and a dense value sweep.
[[nodiscard]] inline auto qwen_rne_nonneg(Var<float> x) noexcept {
    const auto fl = floor(x);
    const auto frac = x - fl;
    const auto fl_i = cast<uint>(fl);
    const auto up = (frac > 0.5f) | ((frac == 0.5f) & ((fl_i & 1u) != 0u));
    return Var<uint>{ite(up, fl_i + 1u, fl_i)};
}

// FP32 -> E4M3FN code, bit-identical to qwen::f32_to_fp8_e4m3 (qwen/common.h):
// RNE, saturation of magnitudes >= 448 to the largest finite code 0x7e, +/-0
// preserved, NaN stored as the positive quiet NaN code 0x7f, +/-inf saturated.
//
// The reference implementation rounds TWICE and so does this one, in the same
// order: first the magnitude is expressed in units of 2^-9 (the subnormal step)
// and rounded to that integer grid, then that integer is rounded to the 3-bit
// E4M3 significand. Rounding the original FP32 value directly would differ
// whenever the discarded bits below 2^-9 tip a mantissa tie (v/scale =
// 12.500343 rounds to 12 under the reference and to 13 under single rounding),
// so the KV cache would store a different code than the weight path for the
// same value. Both steps use RNE, and the 2^-9 scaling is one exact
// power-of-two multiply.
[[nodiscard]] inline auto qwen_fp8_e4m3_from_f32(Var<float> v) noexcept {
    const auto bits = as<uint>(v);
    const auto sign = (bits >> 24u) & 0x80u;
    const auto abs_bits = bits & 0x7FFFFFFFu;
    const auto mag = as<float>(abs_bits);
    const auto is_nan = abs_bits > 0x7F800000u;
    // |v| / 2^-9. Exact for every finite input below 2^121; above that it
    // overflows, so the value is clamped to a representable 2^18 before the
    // integer grid rounding (both ends saturate below).
    const auto scaled_raw = mag * 512.0f;
    const auto scaled = ite(scaled_raw > 262144.0f, 262144.0f, scaled_raw);
    // Round 1: the 2^-9 grid (exactly representable: < 2^19).
    const auto units = qwen_rne_nonneg(scaled);
    Var<uint> code = def(0u);
    $if (units >= 229376u) {
        code = 0x7eu;// 448 * 2^-9: saturate to the largest finite magnitude
    } $else {
        $if (units < 8u) {
            // Subnormal (or exactly 0): the code IS the rounded unit count; 8 units
            // is already the smallest normal code 0x08.
            code = units;
        } $else {
            // Round 2: the 3-bit significand. `p` is the position of the highest set
            // bit of the unit count, so the multiplier 2^(3-p) is an exact power of
            // two and the product keeps every bit of the (integer) unit count.
            const auto units_f = cast<float>(units);
            const auto p = cast<int>((as<uint>(units_f) >> 23u) & 0xFFu) - 127;
            const auto shift = cast<uint>(3 - p);
            const auto mul = as<float>((shift + 127u) << 23u);
            Var<uint> q = qwen_rne_nonneg(units_f * mul);
            Var<int> biased = p - 2;
            $if (q == 16u) {
                q = 8u;// carry into the exponent at the top of a binade
                biased = biased + 1;
            };
            code = ite(biased > 15, 0x7eu,
                       ite(biased <= 0, q, (cast<uint>(biased) << 3u) | (q - 8u)));
        };
    };
    return Var<uint>{ite(is_nan, 0x7fu, sign | code)};
}

// FP8 code of a value stored with `scale`: round_e4m3(v / scale). A non-positive
// scale is the "never written" sentinel, which is only ever combined with values
// that are exactly zero, so the divide is replaced by 0 instead of a 0/0 NaN.
[[nodiscard]] inline auto qwen_fp8_code_scaled(Var<float> v, Var<float> scale) noexcept {
    const auto usable = scale > 0.0f;
    const auto safe_scale = ite(usable, scale, 1.0f);
    return Var<uint>{qwen_fp8_e4m3_from_f32(ite(usable, v / safe_scale, 0.0f))};
}

// The block scale a maximum magnitude implies: max_abs / 448 (0 for an all-zero
// block). 448 is the largest finite E4M3FN magnitude.
[[nodiscard]] inline auto qwen_kv_block_scale_from_max(Var<float> max_abs) noexcept {
    // ONE rounded reciprocal multiply (kQwenKvInvFp8Max), not a division: a
    // fast-math build would lower `/448` into an approximate reciprocal and move the
    // last bit of the scale, breaking bit-equality with the CPU oracle.
    return Var<float>{ite(max_abs > 0.0f, max_abs * kQwenKvInvFp8Max, 0.0f)};
}

// True for a finite value. The append maximum substitutes 448 for non-finite
// values, preserving their codes with a positive scale rather than zeroing it.
[[nodiscard]] inline auto qwen_kv_counts_for_max(Var<float> v) noexcept {
    const auto m = ite(v < 0.0f, -v, v);// |v|, NaN-preserving
    return Var<bool>{m > -3.402823466e38f & m < 3.402823466e38f};
}
// Pack `elements_per_word` codes into one word (LOW code first).
[[nodiscard]] inline auto qwen_kv_pack_word(Var<float> lo, Var<float> hi, bool packed) noexcept {
    if (!packed) { return Var<uint>{as<uint>(lo)}; }
    return Var<uint>{qwen_bf16_from_f32(lo) | (qwen_bf16_from_f32(hi) << 16u)};
}
}// namespace

namespace {
    // Shared body of the fused split + QK norm + partial RoPE + KV append op.
// `base` is the absolute position of token 0: a host kernel argument in the
// reference realization, state[qwen_dec_state_position] in the device-state one.
// `implicit` selects base+t over the per-token positions buffer. Everything else
// is identical, so the two realizations agree element for element.
void qwen_attn_prepare_body(const BufferFloat &q_proj, const BufferFloat &k_proj, const BufferFloat &v_proj,
                             const BufferFloat &q_norm_w, const BufferFloat &k_norm_w, const BufferInt &positions,
                             const BufferFloat &q_out, const BufferFloat &gate_out,
                             const BufferFloat &k_out, const BufferFloat &v_out,
                             const BufferFloat &k_cache, const BufferFloat &v_cache,
                            Var<uint> base,
                            Var<uint> tokens, Var<uint> q_heads, Var<uint> kv_heads,
                            Var<uint> head_dim, Var<uint> rope_dim, Var<float> theta, Var<float> eps,
                            Var<uint> cache_capacity,
                            Var<uint> q_rows, Var<uint> k_rows, Var<uint> v_rows,
                              bool implicit, bool append_kv, QwenKvStorage kv) noexcept {
    set_block_size(128u, 1u, 1u);
    set_warp_size(32u);
    const auto warp = qwen_linear_id() / 32u;
    const auto lane = warp_lane_id();
    const auto total_rows = q_rows + k_rows + v_rows;
    // One warp per (token, head, kind) row; kind order Q, K, V.
    $if (warp < total_rows & head_dim > 0u & tokens > 0u) {
        Var<uint> kind = def(0u);// 0=Q,1=K,2=V
        Var<uint> row = def(warp);
        $if (row >= q_rows) {
            row -= q_rows;
            kind = 1u;
            $if (row >= k_rows) {
                row -= k_rows;
                kind = 2u;
            };
        };
        // row is now (token, head) within its kind.
        Var<uint> heads = q_heads;
        $if (kind != 0u) { heads = kv_heads; };
        const auto t = row / heads;
        const auto h = row % heads;
        const auto pos = implicit ? cast<float>(base + t) : cast<float>(positions.read(t));
        $if (kind == 0u) {
            // Q row: source stride 2*head_dim per head ([Q|gate]).
            const auto src = (t * q_heads + h) * (2u * head_dim);
            const auto dst = (t * q_heads + h) * head_dim;
            Var<float> sumsq = def(0.0f);
            $for (i, lane, head_dim, 32u) {
                const auto v = q_proj.read(src + i);
                sumsq += v * v;
            };
            sumsq = warp_active_sum(sumsq);
            const auto inv = rsqrt(sumsq / cast<float>(head_dim) + eps);
            $for (i, lane, head_dim, 32u) {
                const auto v = q_proj.read(src + i) * inv * q_norm_w.read(i);
                Var<float> out_v = def(v);
                $if (i < rope_dim) {
                    // NeoX rotate-half: pair (p, p+half), half = rope_dim/2.
                    // Each lane recomputes both elements of its pair so it
                    // writes exactly its own element.
                    const auto half = rope_dim / 2u;
                    const auto p = i % half;
                    const auto lo = p;
                    const auto hi = p + half;
                    const auto x0 = q_proj.read(src + lo) * inv * q_norm_w.read(lo);
                    const auto x1 = q_proj.read(src + hi) * inv * q_norm_w.read(hi);
                    const auto rr = qwen_rope_pair(x0, x1, pos, p, rope_dim, theta);
                    out_v = ite(i < half, rr.get<0>(), rr.get<1>());
                };
                q_out.write(dst + i, out_v);
            };
            // Gate: raw copy, no norm/rope.
            $for (i, lane, head_dim, 32u) {
                gate_out.write(dst + i, q_proj.read(src + head_dim + i));
            };
        } $else {
            const auto src = (t * kv_heads + h) * head_dim;
            const auto dst = (t * kv_heads + h) * head_dim;
            $if (kind == 1u) {
                Var<float> sumsq = def(0.0f);
                $for (i, lane, head_dim, 32u) {
                    const auto v = k_proj.read(src + i);
                    sumsq += v * v;
                };
                sumsq = warp_active_sum(sumsq);
                const auto inv = rsqrt(sumsq / cast<float>(head_dim) + eps);
                $for (i, lane, head_dim, 32u) {
                    const auto v = k_proj.read(src + i) * inv * k_norm_w.read(i);
                    Var<float> out_v = def(v);
                    $if (i < rope_dim) {
                        const auto half = rope_dim / 2u;
                        const auto p = i % half;
                        const auto lo = p;
                        const auto hi = p + half;
                        const auto x0 = k_proj.read(src + lo) * inv * k_norm_w.read(lo);
                        const auto x1 = k_proj.read(src + hi) * inv * k_norm_w.read(hi);
                        const auto rr = qwen_rope_pair(x0, x1, pos, p, rope_dim, theta);
                        out_v = ite(i < half, rr.get<0>(), rr.get<1>());
                    };
                    k_out.write(dst + i, out_v);
                            if (append_kv) {
                                // Inline append exists for the FP32 legacy path
                                // only; a packed store needs both codes of a word
                                // in one thread (see QwenKvStorage).
                                const auto slot = qwen_kv_code_index(base + t, h, i, kv_heads, head_dim,
                                                                     cache_capacity, kv.blocked());
                                $if (base + t < cache_capacity) { k_cache.write(slot, out_v); };
                            }
                };
            } $else {
                $for (i, lane, head_dim, 32u) {
                    const auto v = v_proj.read(src + i);
                    v_out.write(dst + i, v);
                            if (append_kv) {
                                const auto slot = qwen_kv_code_index(base + t, h, i, kv_heads, head_dim,
                                                                     cache_capacity, kv.blocked());
                                $if (base + t < cache_capacity) { v_cache.write(slot, v); };
                            }
                };
            };
        };
    };
}

}// namespace

[[nodiscard]] QwenAttnPrepareKernel make_qwen_attn_prepare_kernel(
    bool append_kv, bool implicit_position_ramp, QwenKvStorage kv) noexcept {
    // A packed (bf16) cache is appended through make_qwen_kv_append_packed_kernel:
    // two adjacent channels of one position share a word and this kernel gives them
    // to different lanes, so a per-code store would race.
    LUISA_ASSERT(!append_kv || !kv.packed_words(),
                 "make_qwen_attn_prepare_kernel cannot append a packed KV cache inline");
    QwenAttnPrepareKernel kernel = [append_kv, implicit_position_ramp, kv](
                                       BufferFloat q_proj, BufferFloat k_proj, BufferFloat v_proj,
                                       BufferFloat q_norm_w, BufferFloat k_norm_w, BufferInt positions,
                                       BufferFloat q_out, BufferFloat gate_out, BufferFloat k_out, BufferFloat v_out,
                                       BufferFloat k_cache, BufferFloat v_cache,
                                       Var<uint> tokens, Var<uint> q_heads, Var<uint> kv_heads,
                                       Var<uint> head_dim, Var<uint> rope_dim, Var<float> theta, Var<float> eps,
                                       Var<uint> pos0, Var<uint> cache_capacity,
                                       Var<uint> q_rows, Var<uint> k_rows, Var<uint> v_rows) noexcept {
        set_name(append_kv ? "qwen_attn_prepare_append" : "qwen_attn_prepare");
        qwen_attn_prepare_body(q_proj, k_proj, v_proj, q_norm_w, k_norm_w, positions,
                               q_out, gate_out, k_out, v_out, k_cache, v_cache,
                               pos0, tokens, q_heads, kv_heads, head_dim, rope_dim, theta, eps,
                               cache_capacity, q_rows, k_rows, v_rows,
                             implicit_position_ramp, append_kv, kv);
    };
    return kernel;
}

[[nodiscard]] QwenAttnPrepareDevKernel make_qwen_attn_prepare_dev_kernel(
    bool append_kv, QwenKvStorage kv) noexcept {
    LUISA_ASSERT(!append_kv || !kv.packed_words(),
                 "make_qwen_attn_prepare_dev_kernel cannot append a packed KV cache inline");
    QwenAttnPrepareDevKernel kernel = [append_kv, kv](BufferFloat q_proj, BufferFloat k_proj, BufferFloat v_proj,
                                         BufferFloat q_norm_w, BufferFloat k_norm_w, BufferInt positions,
                                         BufferFloat q_out, BufferFloat gate_out,
                                         BufferFloat k_out, BufferFloat v_out,
                                         BufferFloat k_cache, BufferFloat v_cache, BufferUInt state,
                                         Var<uint> tokens, Var<uint> q_heads, Var<uint> kv_heads,
                                         Var<uint> head_dim, Var<uint> rope_dim, Var<float> theta, Var<float> eps,
                                         Var<uint> cache_capacity,
                                         Var<uint> q_rows, Var<uint> k_rows, Var<uint> v_rows) noexcept {
        set_name("qwen_attn_prepare_dev");
        // A decode step always writes its own K/V at the device-resident
        // position, so append is unconditional and the ramp is implicit. The
        // positions buffer stays bound (the caller passes the id staging view)
        // and is never read.
        qwen_attn_prepare_body(q_proj, k_proj, v_proj, q_norm_w, k_norm_w, positions,
                               q_out, gate_out, k_out, v_out, k_cache, v_cache,
                               state.read(qwen_dec_state_position),
                               tokens, q_heads, kv_heads, head_dim, rope_dim, theta, eps,
                               cache_capacity, q_rows, k_rows, v_rows, true, append_kv, kv);
    };
    return kernel;
}

// ---------------------------------------------------------------------------
// Model-specialized decode attention prepare (opt_plan/09 N3.2). See dsl_ops.h.
// ---------------------------------------------------------------------------
namespace {
// Shared body of the model-specialized prepare. `base` is the absolute position of
// token 0: a host kernel argument in the reference realization,
// state[qwen_dec_state_position] in the device-state one. Compile-time geometry:
// head_dim 256, rope_dim 64, 16 q heads, 4 kv heads.
void qwen35_attn_prepare_model_body(
    const BufferFloat &q_proj, const BufferFloat &k_proj, const BufferFloat &v_proj,
    const BufferFloat &q_norm_w, const BufferFloat &k_norm_w,
    const BufferFloat &q_out, const BufferFloat &gate_out,
    const BufferFloat &k_out, const BufferFloat &v_out,
    const BufferFloat &k_cache, const BufferFloat &v_cache,
    Var<uint> base, Var<uint> tokens, Var<float> theta, Var<float> eps,
      Var<uint> cache_capacity, bool append_kv, QwenKvStorage kv) noexcept {
    constexpr uint32_t kHD = qwen35_attn_model_head_dim;
    constexpr uint32_t kRope = qwen35_attn_model_rope_dim;
    constexpr uint32_t kQH = qwen35_attn_model_q_heads;
    constexpr uint32_t kKVH = qwen35_attn_model_kv_heads;
    constexpr uint32_t kLanes = kHD / 32u;// eight constant-indexed elements per lane
    // The rotated prefix is exactly one NeoX pair per lane (rope_dim/2 == 32).
    static_assert(kRope == 2u * 32u, "rope prefix must be exactly one NeoX pair per lane");
    set_block_size(128u, 1u, 1u);
    set_warp_size(32u);
    const auto warp = qwen_linear_id() / 32u;
    const auto lane = warp_lane_id();
    const auto q_rows = tokens * kQH;
    const auto kv_rows = tokens * kKVH;
    $if (warp < q_rows + 2u * kv_rows & tokens > 0u) {
        // kind order Q, K, V -- identical to the generic kernel.
        Var<uint> kind = def(0u);
        Var<uint> row = def(warp);
        $if (row >= q_rows) {
            row -= q_rows;
            kind = 1u;
            $if (row >= kv_rows) {
                row -= kv_rows;
                kind = 2u;
            };
        };
        Var<uint> heads = def(kQH);
        $if (kind != 0u) { heads = kKVH; };
        const auto t = row / heads;
        const auto h = row % heads;
        const auto pos = cast<float>(base + t);
        const auto dst = row * kHD;
        // Cache the lane's eight elements and its eight norm weights, and reduce.
        const auto norm_pass = [&](const BufferVar<float> &src_buf, Var<uint> src,
                                   const BufferVar<float> &wbuf, Local<float> &cache,
                                   Local<float> &wcache) {
            Var<float> sumsq = def(0.0f);
            for (uint32_t j = 0u; j < kLanes; j++) {
                cache[j] = src_buf.read(src + lane + 32u * j);
                wcache[j] = wbuf.read(lane + 32u * j);
                sumsq += cache[j] * cache[j];
            }
            return warp_active_sum(sumsq);
        };
        $if (kind == 0u) {
            // Q row: source stride 2*head_dim per head ([Q|gate]).
            const auto src = (t * kQH + h) * (2u * kHD);
            Local<float> cache{kLanes};
            Local<float> wcache{kLanes};
            const auto sumsq = norm_pass(q_proj, src, q_norm_w, cache, wcache);
            const auto inv = rsqrt(sumsq / static_cast<float>(kHD) + eps);
            // The rotated prefix is exactly pairs (lane, lane+32): one trig
            // evaluation per lane instead of two.
            const auto x0 = cache[0] * inv * wcache[0];
            const auto x1 = cache[1] * inv * wcache[1];
            const auto rr = qwen_rope_pair(x0, x1, pos, lane, kRope, theta);
            q_out.write(dst + lane, rr.get<0>());
            q_out.write(dst + lane + 32u, rr.get<1>());
            for (uint32_t j = 2u; j < kLanes; j++) {
                q_out.write(dst + lane + 32u * j, cache[j] * inv * wcache[j]);
            }
            // Gate: raw copy of the second half, exactly like the generic kernel.
            for (uint32_t j = 0u; j < kLanes; j++) {
                gate_out.write(dst + lane + 32u * j,
                               q_proj.read(src + kHD + lane + 32u * j));
            }
        } $elif (kind == 1u) {
            const auto src = row * kHD;
            Local<float> cache{kLanes};
            Local<float> wcache{kLanes};
            const auto sumsq = norm_pass(k_proj, src, k_norm_w, cache, wcache);
            const auto inv = rsqrt(sumsq / static_cast<float>(kHD) + eps);
            const auto x0 = cache[0] * inv * wcache[0];
            const auto x1 = cache[1] * inv * wcache[1];
            const auto rr = qwen_rope_pair(x0, x1, pos, lane, kRope, theta);
              // Cache slot of this (token, head, channel 0): the dtype/layout
              // contract decides whether that is a legacy row or a 64-position
              // block (see qwen_kv_code_index).
              const auto slot0 = qwen_kv_code_index(base + t, h, 0u, kKVH, kHD, cache_capacity,
                                                    kv.blocked());
              Var<float> v0 = rr.get<0>();
              Var<float> v1 = rr.get<1>();
              k_out.write(dst + lane, v0);
              k_out.write(dst + lane + 32u, v1);
              if (append_kv) {
                  $if (base + t < cache_capacity) {
                      k_cache.write(slot0 + lane, v0);
                      k_cache.write(slot0 + lane + 32u, v1);
                  };
              }
              for (uint32_t j = 2u; j < kLanes; j++) {
                  const auto v = cache[j] * inv * wcache[j];
                  k_out.write(dst + lane + 32u * j, v);
                  if (append_kv) {
                      $if (base + t < cache_capacity) {
                          k_cache.write(slot0 + lane + 32u * j, v);
                      };
                  }
              }
        } $else {
              const auto src = row * kHD;
              const auto slot0 = qwen_kv_code_index(base + t, h, 0u, kKVH, kHD, cache_capacity,
                                                    kv.blocked());
              for (uint32_t j = 0u; j < kLanes; j++) {
                  const auto v = v_proj.read(src + lane + 32u * j);
                  v_out.write(dst + lane + 32u * j, v);
                  if (append_kv) {
                      $if (base + t < cache_capacity) {
                          v_cache.write(slot0 + lane + 32u * j, v);
                      };
                  }
              }
        };
    };
}
}// namespace

[[nodiscard]] QwenAttnPrepareModelKernel make_qwen35_attn_prepare_model_kernel(
    bool append_kv, QwenKvStorage kv) noexcept {
    LUISA_ASSERT(!append_kv || !kv.packed_words(),
                 "make_qwen35_attn_prepare_model_kernel cannot append a packed KV cache inline");
    QwenAttnPrepareModelKernel kernel = [append_kv, kv](
                                            BufferFloat q_proj, BufferFloat k_proj, BufferFloat v_proj,
                                            BufferFloat q_norm_w, BufferFloat k_norm_w,
                                            BufferFloat q_out, BufferFloat gate_out,
                                            BufferFloat k_out, BufferFloat v_out,
                                            BufferFloat k_cache, BufferFloat v_cache,
                                            Var<uint> tokens, Var<float> theta, Var<float> eps,
                                            Var<uint> pos0, Var<uint> cache_capacity) noexcept {
        set_name("qwen35_attn_prepare_model");
        qwen35_attn_prepare_model_body(q_proj, k_proj, v_proj, q_norm_w, k_norm_w,
                                       q_out, gate_out, k_out, v_out, k_cache, v_cache,
                             pos0, tokens, theta, eps, cache_capacity, append_kv, kv);
    };
    return kernel;
}

[[nodiscard]] QwenAttnPrepareModelDevKernel make_qwen35_attn_prepare_model_dev_kernel(
    bool append_kv, QwenKvStorage kv) noexcept {
    LUISA_ASSERT(!append_kv || !kv.packed_words(),
                 "make_qwen35_attn_prepare_model_dev_kernel cannot append a packed KV cache inline");
    QwenAttnPrepareModelDevKernel kernel = [append_kv, kv](
                                               BufferFloat q_proj, BufferFloat k_proj, BufferFloat v_proj,
                                               BufferFloat q_norm_w, BufferFloat k_norm_w,
                                               BufferFloat q_out, BufferFloat gate_out,
                                               BufferFloat k_out, BufferFloat v_out,
                                               BufferFloat k_cache, BufferFloat v_cache,
                                               BufferUInt state,
                                               Var<uint> tokens, Var<float> theta, Var<float> eps,
                                               Var<uint> cache_capacity) noexcept {
        set_name("qwen35_attn_prepare_model_dev");
        qwen35_attn_prepare_model_body(q_proj, k_proj, v_proj, q_norm_w, k_norm_w,
                                       q_out, gate_out, k_out, v_out, k_cache, v_cache,
                                 state.read(qwen_dec_state_position),
                                 tokens, theta, eps, cache_capacity, append_kv, kv);
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_attn_prepare_model_dispatch(uint64_t tokens) {
    const uint64_t rows = tokens * (static_cast<uint64_t>(qwen35_attn_model_q_heads) +
                                    2u * qwen35_attn_model_kv_heads);
    return plan_warp_rows(rows, 128u);
}

[[nodiscard]] Status validate_qwen35_attn_prepare_model_inputs(
    uint64_t tokens, const QwenDslConfig &cfg, uint64_t pos0, uint64_t cache_capacity) {
    if (auto s = validate_qwen_config(cfg); !s.ok()) { return s; }
    if (cfg.attn_head_dim != qwen35_attn_model_head_dim ||
        cfg.rope_dim != qwen35_attn_model_rope_dim ||
        cfg.q_heads != qwen35_attn_model_q_heads ||
        cfg.kv_heads != qwen35_attn_model_kv_heads) {
        return Status::error(StatusCode::unsupported,
                             "model attention prepare needs head_dim 256 / rope_dim 64 / 16 q / 4 kv heads",
                             luisa::format("head_dim={} rope_dim={} q_heads={} kv_heads={}",
                                           cfg.attn_head_dim, cfg.rope_dim,
                                           cfg.q_heads, cfg.kv_heads));
    }
    if (tokens == 0u) { return Status::success(); }
    if (tokens > 0xFFFFFFFFull) { return err_overflow("attention tokens exceed uint32"); }
    if (!checked_product_u32(tokens, cfg.attn_q_proj_n(), "attention q projection").ok() ||
        !checked_product_u32(tokens, cfg.attn_kv_n(), "attention kv projection").ok()) {
        return err_overflow("attention projection element count exceeds uint32");
    }
    if (pos0 > 0x7FFFFFFFull || tokens > 0x7FFFFFFFull - pos0) {
        return err_overflow("attention absolute position exceeds int32");
    }
    if (pos0 + tokens > cache_capacity) {
        return err_invalid("attention KV append would exceed cache capacity");
    }
    if (!checked_product_u32(cache_capacity, cfg.attn_kv_n(), "KV cache").ok()) {
        return err_overflow("KV cache element count exceeds uint32");
    }
    return Status::success();
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_prepare_dispatch(uint64_t tokens, const QwenDslConfig &cfg) {
    const uint64_t rows = tokens * (static_cast<uint64_t>(cfg.q_heads) + 2u * cfg.kv_heads);
    return plan_warp_rows(rows, 128u);
}

[[nodiscard]] Status validate_qwen_attn_prepare_inputs(uint64_t tokens, const QwenDslConfig &cfg,
                                                          uint64_t pos0, uint64_t cache_capacity, bool append_kv) {
    const auto cfg_ok = validate_qwen_config(cfg);
    if (!cfg_ok.ok()) { return cfg_ok; }
    if (tokens == 0u) { return Status::success(); }
    if (tokens > 0xFFFFFFFFull) { return err_overflow("attention tokens exceed uint32"); }
    if (!checked_product_u32(tokens, cfg.attn_q_proj_n(), "attention q projection").ok() ||
        !checked_product_u32(tokens, cfg.attn_kv_n(), "attention kv projection").ok()) {
        return err_overflow("attention projection element count exceeds uint32");
    }
    if (pos0 > 0x7FFFFFFFull || tokens > 0x7FFFFFFFull - pos0) {
        return err_overflow("attention absolute position exceeds int32");
    }
    if (append_kv) {
        if (pos0 + tokens > cache_capacity) {
            return err_invalid("attention KV append would exceed cache capacity");
        }
        if (!checked_product_u32(cache_capacity, cfg.attn_kv_n(), "KV cache").ok()) {
            return err_overflow("KV cache element count exceeds uint32");
        }
    }
    return Status::success();
}

[[nodiscard]] QwenRopePartialKernel make_qwen_rope_partial_neox_kernel() noexcept {
    QwenRopePartialKernel kernel = [](BufferFloat x, BufferFloat out, BufferInt positions,
                                      Var<uint> tokens, Var<uint> heads, Var<uint> head_dim,
                                      Var<uint> rope_dim, Var<float> theta, Var<uint> rows) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name("qwen_rope_partial_neox");
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < rows & head_dim > 0u & tokens > 0u) {
            const auto t = warp / heads;
            const auto base = warp * head_dim;
            const auto pos = cast<float>(positions.read(t));
            // Rotated prefix: NeoX pairs (p, p+half), half = rope_dim/2;
            // each lane handles pairs p = lane, lane+32, ... and writes both
            // elements of its pair exclusively.
            const auto half = rope_dim / 2u;
            $for (p, lane, half, 32u) {
                const auto x0 = x.read(base + p);
                const auto x1 = x.read(base + p + half);
                const auto rr = qwen_rope_pair(x0, x1, pos, p, rope_dim, theta);
                out.write(base + p, rr.get<0>());
                out.write(base + p + half, rr.get<1>());
            };
            // Tail passthrough.
            $for (i, rope_dim + lane, head_dim, 32u) {
                out.write(base + i, x.read(base + i));
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_rope_dispatch(uint64_t tokens, uint64_t heads) {
    const auto rows = checked_product_u32(tokens, heads, "rope rows");
    if (!rows.ok()) { return Result<QwenLaunch1D>::failure(rows.status); }
    return plan_warp_rows(*rows.value, 128u);
}

[[nodiscard]] QwenKvAppendKernel make_qwen_kv_append_kernel() noexcept {
    QwenKvAppendKernel kernel = [](BufferFloat k, BufferFloat v, BufferFloat k_cache, BufferFloat v_cache,
                                   Var<uint> tokens, Var<uint> channels, Var<uint> pos0,
                                   Var<uint> capacity, Var<uint> padded_cols) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("qwen_kv_append");
        const auto linear = qwen_linear_id();
        const auto row = linear / padded_cols;
        const auto col = linear % padded_cols;
        $if (row < tokens & col < channels) {
            const auto p = pos0 + row;
            $if (p < capacity) {
                const auto src = row * channels + col;
                const auto dst = p * channels + col;
                k_cache.write(dst, k.read(src));
                v_cache.write(dst, v.read(src));
            };
        };
    };
    return kernel;
}

// ---------------------------------------------------------------------------
// Packed KV append (BF16 KV cache). One thread per (token, kv_head, WORD): both
// codes of the word are produced by that thread from the source row, so there is no
// read-modify-write and no cross-lane race. At f32 (elements_per_word == 1) this is
// the legacy dense append with the same addressing contract.
// ---------------------------------------------------------------------------
namespace {
void qwen_kv_append_packed_body(const BufferFloat &k, const BufferFloat &v,
                                const BufferUInt &k_cache, const BufferUInt &v_cache,
                                Var<uint> base, Var<uint> tokens, Var<uint> kv_heads,
                                Var<uint> head_dim, Var<uint> cache_capacity,
                                QwenKvStorage kv) noexcept {
    const uint32_t epw = kv.elements_per_word();
    const uint32_t blocked = kv.blocked() ? 1u : 0u;
    set_block_size(128u, 1u, 1u);
    const auto words_per_head = head_dim / epw;
    const auto width = kv_heads * words_per_head;
    const auto linear = qwen_linear_id();
    const auto total = tokens * width;
    $if (linear < total) {
        const auto w = linear % words_per_head;
        const auto row = linear / words_per_head;
        const auto h = row % kv_heads;
        const auto t = row / kv_heads;
        const auto pos = base + t;
        const auto channel = w * epw;
        $if (pos < cache_capacity) {
            const auto dst = qwen_kv_code_index(pos, h, channel, kv_heads, head_dim,
                                                cache_capacity, blocked != 0u);
            const auto src = row * head_dim + channel;
            if (epw == 1u) {
                k_cache.write(dst, as<uint>(k.read(src)));
                v_cache.write(dst, as<uint>(v.read(src)));
            } else {
                k_cache.write(dst / epw,
                              qwen_kv_pack_word(k.read(src), k.read(src + 1u), true));
                v_cache.write(dst / epw,
                              qwen_kv_pack_word(v.read(src), v.read(src + 1u), true));
            }
        };
    };
}
}// namespace

[[nodiscard]] QwenKvAppendPackedKernel make_qwen_kv_append_packed_kernel(
    QwenKvStorage kv, bool device_state) noexcept {
    QwenKvAppendPackedKernel kernel = [kv, device_state](
                                          BufferFloat k, BufferFloat v,
                                          BufferUInt k_cache, BufferUInt v_cache, BufferUInt state,
                                          Var<uint> tokens, Var<uint> kv_heads, Var<uint> head_dim,
                                          Var<uint> pos0, Var<uint> cache_capacity) noexcept {
        set_name(device_state ? "qwen_kv_append_packed_dev" : "qwen_kv_append_packed");
        // The device-state realization takes the absolute position of token 0 from
        // state[position] instead of a host scalar, which is what keeps a captured
        // CUDA graph argument-invariant (opt_plan/03 section 2.3).
        const auto base = device_state ? state.read(qwen_dec_state_position) : pos0;
        qwen_kv_append_packed_body(k, v, k_cache, v_cache, base, tokens, kv_heads, head_dim,
                                   cache_capacity, kv);
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_kv_append_packed_dispatch(
    uint64_t tokens, uint64_t kv_heads, uint64_t head_dim, QwenKvStorage kv) {
    if (head_dim == 0u || kv_heads == 0u) {
        return Result<QwenLaunch1D>::failure(err_invalid("kv append requires a nonzero geometry"));
    }


    if (head_dim % kv.elements_per_word() != 0u) {
        return Result<QwenLaunch1D>::failure(
            err_invalid("kv append requires head_dim to be a whole number of stored words"));
    }
    const uint64_t words_per_head = head_dim / kv.elements_per_word();
    const uint64_t n = tokens * kv_heads * words_per_head;
    if (tokens != 0u && n / tokens / kv_heads != words_per_head) {
        return Result<QwenLaunch1D>::failure(err_overflow("kv append word count overflows"));
    }
    if (n > 0xFFFFFFFFull) {
        return Result<QwenLaunch1D>::failure(err_overflow("kv append word count exceeds uint32"));
    }
    auto plan = plan_qwen_threads(static_cast<uint32_t>(n), 128u);
    if (!plan.ok()) { return Result<QwenLaunch1D>::failure(plan.status); }
    // work_items is the number of WORDS, not the number of 128-thread blocks;
    // dispatch totals are already ceil-divided by plan_qwen_threads().
    return plan;
}

// ---------------------------------------------------------------------------
// FP8 E4M3 KV append with ONLINE per-(kv_head, 64-position block) scales.
//
// The CPU oracle of this exact rule is qwen_kv_append_reference
// (qwen/kv_cache_layout.cpp): the device test compares the resulting words and
// scales bit for bit, so the two implementations are one contract.
//
// One 128-thread CTA per (touched 64-position block, kv_head). K then V:
//   1. every thread reduces max|v| over its words of the NEW rows (non-finite
//      values excluded), warp_active_max folds the warp, the first warp's lanes
//      publish one value each and thread 0 folds those;
//   2. thread 0 reads the block's current scale, computes
//      new = max(old, max_abs/448) and stores it: `grow` is exactly
//      "max_abs > 448*old", so a block whose maximum does not grow keeps its
//      scale AND every earlier code byte-identical;
//   3. a grown block rewrites its whole valid prefix (codes dequantized with the
//      OLD scale, re-encoded against the NEW one); otherwise only the new rows
//      are written. Each thread owns whole WORDS (four channels), so no
//      read-modify-write crosses lanes, and the scale store is separated from the
//      rewrite by the barrier.
// ---------------------------------------------------------------------------
[[nodiscard]] uint32_t qwen_kv_append_fp8_shared_bytes() noexcept {
    // one block maximum per warp (4 warps) + the per-kind decision record
    return (4u + 12u) * static_cast<uint32_t>(sizeof(float));
}

namespace {
// Blocked64 code index of (head, block, position-in-block, channel), i.e. the same
// arithmetic as QwenKvLayoutDesc::code_index with the position split so the block
// index is not recomputed from the position.
[[nodiscard]] inline auto qwen_kv_blocked_code(Var<uint> head, Var<uint> block,
                                               Var<uint> pos_in_block, Var<uint> channel,
                                               Var<uint> head_dim, Var<uint> block_count) noexcept {
    return Var<uint>{((head * block_count + block) * qwen_kv_position_block + pos_in_block) *
                         head_dim +
                     channel};
}

// One (kv_head, 64-position block) of ONE region (K or V). `new_begin`/`new_end`
// are BLOCK-LOCAL positions: [new_begin, new_end) are the rows this append adds,
// [0, new_begin) are already stored, [new_end, 64) are not valid yet.
void qwen_kv_append_fp8_region(const BufferFloat &src, const BufferUInt &cache,
                               const BufferFloat &scale_buf,
                               Var<uint> head, Var<uint> block,
                               Var<uint> new_begin, Var<uint> new_end,
                               Var<uint> base_pos, Var<uint> head_dim, Var<uint> kv_heads,
                               Var<uint> block_count, Shared<float> &warp_max,
                               Shared<float> &ctl, uint32_t ctl_base) noexcept {
    const auto row_words = head_dim / 4u;// 4 E4M3 codes per word, head_dim % 4 == 0
    const auto scale_slot = head * block_count + block;
    // ---- phase 1: block maximum over the NEW rows -------------------------
    Var<float> local = def(0.0f);
    $for (i, new_begin * row_words + thread_x(), new_end * row_words, 128u) {
        const auto pos_in_block = i / row_words;
        const auto c = (i % row_words) * 4u;
        const auto row = block * qwen_kv_position_block + pos_in_block - base_pos;
        const auto base = (row * kv_heads + head) * head_dim + c;
        for (uint32_t j = 0u; j < 4u; j++) {
            const auto value = src.read(base + j);
            const auto magnitude = ite(value < 0.0f, -value, value);
                                        // A non-finite value contributes one full-scale unit,
                            // so even an inf/NaN-only block keeps a positive scale
                            // and its invalid code instead of becoming a zero.
                            local = max(local, ite(qwen_kv_counts_for_max(value), magnitude,
                                                   static_cast<float>(kFp8E4M3Max)));
        }
    };
    const auto warp_value = warp_active_max(local);
    $if ((thread_x() % 32u) == 0u) { warp_max.write(thread_x() / 32u, warp_value); };
    sync_block();
    $if (thread_x() == 0u) {
        // Deterministic fold over the four warps (empty warps contribute 0).
        Var<float> block_max = def(0.0f);
        for (uint32_t w = 0u; w < 4u; w++) { block_max = max(block_max, warp_max.read(w)); }
        const auto old_scale = scale_buf.read(scale_slot);
        const auto new_scale = max(old_scale, qwen_kv_block_scale_from_max(block_max));
        scale_buf.write(scale_slot, new_scale);
        ctl.write(ctl_base + 0u, new_scale);
        ctl.write(ctl_base + 1u, old_scale);
        ctl.write(ctl_base + 2u, ite(new_scale > old_scale, 1.0f, 0.0f));
    };
    sync_block();
    const auto scale_new = ctl.read(ctl_base + 0u);
    const auto scale_old = ctl.read(ctl_base + 1u);
    const auto grows = ctl.read(ctl_base + 2u) > 0.5f;
    // ---- phase 2: write the new rows (and the prefix if the scale grew) ----
    const auto first_word = ite(grows, 0u, new_begin * row_words);
    $for (i, first_word + thread_x(), new_end * row_words, 128u) {
        const auto pos_in_block = i / row_words;
        const auto c = (i % row_words) * 4u;
        const auto code = qwen_kv_blocked_code(head, block, pos_in_block, c, head_dim, block_count);
        Var<uint> word = def(0u);
        $if (pos_in_block < new_begin) {
            // Stored earlier: re-encode the SAME value against the new scale.
            const auto old_word = cache.read(code / 4u);
            for (uint32_t j = 0u; j < 4u; j++) {
                const auto value =
                    qwen_fp8_e4m3_to_f32_branchless((old_word >> (8u * j)) & 0xFFu) * scale_old;
                word = word | (qwen_fp8_code_scaled(value, scale_new) << (8u * j));
            }
        } $else {
            const auto row = block * qwen_kv_position_block + pos_in_block - base_pos;
            const auto base = (row * kv_heads + head) * head_dim + c;
            for (uint32_t j = 0u; j < 4u; j++) {
                word = word | (qwen_fp8_code_scaled(src.read(base + j), scale_new) << (8u * j));
            }
        };
        cache.write(code / 4u, word);
    };
    sync_block();
}
}// namespace

[[nodiscard]] QwenKvAppendFp8Kernel make_qwen_kv_append_fp8_kernel(
    QwenKvStorage kv, bool device_state) noexcept {
    LUISA_ASSERT(kv.scaled() && kv.blocked(),
                 "make_qwen_kv_append_fp8_kernel requires the fp8_e4m3 + blocked64 storage");
    QwenKvAppendFp8Kernel kernel = [device_state](BufferFloat k, BufferFloat v,
                                                  BufferUInt k_cache, BufferUInt v_cache,
                                                  BufferFloat k_scale, BufferFloat v_scale,
                                                  BufferUInt state, Var<uint> tokens,
                                                  Var<uint> kv_heads, Var<uint> head_dim,
                                                  Var<uint> pos0, Var<uint> cache_capacity) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(device_state ? "qwen_kv_append_fp8_dev" : "qwen_kv_append_fp8");
        Shared<float> warp_max{4u};
        Shared<float> ctl{12u};// [0..2] = K decision, [6..8] = V decision
        const auto base_pos = ite(device_state, state.read(qwen_dec_state_position), pos0);
        // Touched blocks of the batch: the first one may start mid-block.
        const auto block_count = (cache_capacity + (qwen_kv_position_block - 1u)) /
                                 qwen_kv_position_block;
        const auto touched = (base_pos % qwen_kv_position_block + tokens +
                              (qwen_kv_position_block - 1u)) /
                             qwen_kv_position_block;
        const auto block_id = qwen_linear_id() / 128u;
        $if (block_id < touched * kv_heads & tokens > 0u & head_dim > 0u & kv_heads > 0u) {
            const auto head = block_id % kv_heads;
            const auto block = base_pos / qwen_kv_position_block + block_id / kv_heads;
            const auto block_start = block * qwen_kv_position_block;
            // ite() rather than max()/min() on two uint Vars: the unqualified
            // std::max/std::min are also visible here and would need operator<.
            const auto new_begin = ite(base_pos > block_start, base_pos, block_start) - block_start;
            const auto end_pos = ite(base_pos + tokens < block_start + qwen_kv_position_block,
                                     base_pos + tokens,
                                     block_start + qwen_kv_position_block);
            const auto new_end = end_pos - block_start;
            $if (new_end > new_begin) {
                qwen_kv_append_fp8_region(k, k_cache, k_scale, head, block, new_begin, new_end,
                                          base_pos, head_dim, kv_heads, block_count, warp_max,
                                          ctl, 0u);
                qwen_kv_append_fp8_region(v, v_cache, v_scale, head, block, new_begin, new_end,
                                          base_pos, head_dim, kv_heads, block_count, warp_max,
                                          ctl, 6u);
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_kv_append_fp8_dispatch(
    uint64_t pos0, uint64_t tokens, uint64_t kv_heads, QwenKvStorage kv) {
    if (!kv.scaled() || !kv.blocked()) {
        return Result<QwenLaunch1D>::failure(
            err_invalid("the fp8 online-scale append requires the fp8_e4m3 + blocked64 storage"));
    }
    if (kv_heads == 0u) {
        return Result<QwenLaunch1D>::failure(err_invalid("KV append needs kv_heads >= 1"));
    }
    if (tokens == 0u) { return Result<QwenLaunch1D>::success(QwenLaunch1D{}); }
    if (pos0 > 0xFFFFFFFFull || tokens > 0xFFFFFFFFull - pos0) {
        return Result<QwenLaunch1D>::failure(
            err_overflow("KV append position exceeds uint32"));
    }
    const uint64_t touched = (pos0 % qwen_kv_position_block + tokens +
                              (qwen_kv_position_block - 1u)) /
                             qwen_kv_position_block;
    const auto blocks = checked_product_u32(touched, kv_heads, "fp8 kv append blocks");
    if (!blocks.ok()) { return Result<QwenLaunch1D>::failure(blocks.status); }
    const auto threads = checked_product_u32(*blocks.value, 128u, "fp8 kv append threads");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto planned = plan_qwen_threads(*threads.value, 128u);
    if (!planned.ok()) { return Result<QwenLaunch1D>::failure(planned.status); }
    planned.value->work_items = *blocks.value;
    planned.value->warp_size = 32u;
    return planned;
}

[[nodiscard]] QwenKvFp8EncodeKernel make_qwen_kv_fp8_encode_kernel() noexcept {
    QwenKvFp8EncodeKernel kernel = [](BufferFloat values, BufferUInt codes,
                                      Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_kv_fp8_encode");
        const auto i = qwen_linear_id();
        $if (i < count) { codes.write(i, qwen_fp8_e4m3_from_f32(values.read(i))); };
    };
    return kernel;
}

[[nodiscard]] QwenKvFp8DecodeKernel make_qwen_kv_fp8_decode_kernel() noexcept {
    QwenKvFp8DecodeKernel kernel = [](BufferUInt codes, BufferFloat values,
                                      Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_kv_fp8_decode");
        const auto i = qwen_linear_id();
        $if (i < count) { values.write(i, qwen_fp8_e4m3_to_f32_branchless(codes.read(i) & 0xFFu)); };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_kv_fp8_codec_dispatch(uint64_t count) {
    if (count == 0u) { return Result<QwenLaunch1D>::success(QwenLaunch1D{}); }
    if (count > 0xFFFFFFFFull) {
        return Result<QwenLaunch1D>::failure(err_overflow("fp8 codec element count exceeds uint32"));
    }
    return plan_qwen_threads(static_cast<uint32_t>(count), 256u);
}


[[nodiscard]] Result<uint64_t> qwen_kv_region_words(QwenKvStorage kv, uint32_t max_context,
                                                     uint32_t kv_heads, uint32_t head_dim) noexcept {
    QwenKvLayoutDesc desc{};
    if (auto s = qwen_kv_layout_desc(kv.dtype, kv.layout, max_context, kv_heads, head_dim, desc);
        !s.ok()) {
        return Result<uint64_t>::failure(s);
    }
    return Result<uint64_t>::success(desc.region_words);
}

[[nodiscard]] Result<QwenLaunch2D> plan_qwen_kv_append_dispatch(uint64_t tokens, uint64_t channels) {
    const auto n = checked_product_u32(tokens, channels, "kv append");
    if (!n.ok()) { return Result<QwenLaunch2D>::failure(n.status); }
    return plan_qwen_threads_2d(tokens, channels);
}

[[nodiscard]] QwenAttnPrefillKernel make_qwen_attn_prefill_kernel(QwenKvStorage kv) noexcept {
    QwenAttnPrefillKernel kernel = [kv](BufferFloat q, BufferFloat gate, BufferFloat k_cache, BufferFloat v_cache,
                                        BufferFloat k_scale, BufferFloat v_scale,
                                        BufferFloat out, Var<uint> tokens, Var<uint> q_heads, Var<uint> kv_heads,
                                        Var<uint> head_dim, Var<float> scale, Var<uint> pos0, Var<uint> kv_len,
                                        Var<uint> capacity) noexcept {
        set_block_size(128u, 1u, 1u);
        set_name("qwen_attn_prefill");
        const auto id = qwen_linear_id();
        $if (id < tokens * q_heads & head_dim > 0u & kv_heads > 0u) {
            const auto t = id / q_heads;
            const auto h = id % q_heads;
            const auto kvh = h / (q_heads / kv_heads);
            const auto q_base = (t * q_heads + h) * head_dim;
            const auto k_end = pos0 + t + 1u;// causal: keys [0, pos0+t]
            // Online softmax: running max m, sum l, output acc[d].
            Var<float> run_m = def(-3.0e38f);
            Var<float> run_l = def(0.0f);
            // Online softmax: running max m, sum l, output acc[d]. The
            // accumulator is a fixed-size local array (max head_dim 1024);
            // only [0, head_dim) is used, explicitly zeroed here.
            Local<float> acc{1024u};
            $for (d, head_dim) { acc[d] = 0.0f; };
            $for (key, k_end) {
                Var<float> score = def(0.0f);
                $for (d, head_dim) {
                    score += q.read(q_base + d) * qwen_kv_load_at(k_cache, k_scale, key, kvh, d, kv_heads,
                                                                  head_dim, capacity, kv);
                };
                score *= scale;
                const auto new_m = max(run_m, score);
                const auto corr = exp(run_m - new_m);
                const auto p = exp(score - new_m);
                run_l = run_l * corr + p;
                $for (d, head_dim) {
                    acc[d] = acc[d] * corr +
                             p * qwen_kv_load_at(v_cache, v_scale, key, kvh, d, kv_heads, head_dim,
                                                 capacity, kv);
                };
                run_m = new_m;
            };
            const auto inv_l = 1.0f / run_l;
            $for (d, head_dim) {
                const auto g = gate.read(q_base + d);
                out.write(q_base + d, acc[d] * inv_l * qwen_sigmoid(g));
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_prefill_dispatch(uint64_t tokens, uint64_t q_heads) {
    const auto n = checked_product_u32(tokens, q_heads, "attention prefill rows");
    if (!n.ok()) { return Result<QwenLaunch1D>::failure(n.status); }
    return plan_qwen_threads(*n.value, 128u);
}

[[nodiscard]] Status validate_qwen_attn_prefill_inputs(uint64_t tokens, uint64_t q_heads, uint64_t kv_heads,
                                                          uint64_t head_dim, uint64_t pos0, uint64_t kv_len,
                                                          uint64_t capacity) {
    if (q_heads == 0u || kv_heads == 0u || q_heads % kv_heads != 0u) {
        return err_invalid("attention GQA requires q_heads % kv_heads == 0");
    }
    if (head_dim == 0u || head_dim > 1024u) {
        return err_invalid("attention head_dim must be in [1,1024] (local accumulator bound)");
    }
    if (tokens == 0u) { return Status::success(); }
    if (pos0 > kv_len || tokens > kv_len - pos0) {
        return err_invalid("attention prefill causal window exceeds kv_len (append first)");
    }
    if (kv_len > capacity) { return err_invalid("attention kv_len exceeds cache capacity"); }
    const auto cache_n = checked_product_u32(capacity, kv_heads, "attention cache");
    const auto q_n = checked_product_u32(tokens, q_heads, "attention queries");
    if (!cache_n.ok() || !q_n.ok() ||
        !checked_product_u32(*cache_n.value, head_dim, "attention cache").ok() ||
        !checked_product_u32(*q_n.value, head_dim, "attention queries").ok()) {
        return err_overflow("attention element count exceeds uint32");
    }
    if (pos0 > 0xFFFFFFFFull || tokens > 0xFFFFFFFFull - pos0) {
        return err_overflow("attention position exceeds uint32");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Tiled causal GQA prefill attention (model path). See dsl_ops.h for the design.
// ---------------------------------------------------------------------------
namespace {
// NB: the warp reduction used below is qwen_warp_butterfly_sum, defined next to
// the weight decode helpers so the prefill GEMM and this kernel share exactly
// one implementation.
// exp() as exp2(x * log2(e)): exp2 lowers to a single MUFU/OpExtInst
// instruction on every backend, while an IEEE-accurate exp() is a ~20
// instruction range reduction. The only extra rounding is one FP32 multiply,
// orders of magnitude inside the softmax tolerance the differential suite
// enforces (and the reference kernel's exp() is itself compared against a
// double-accumulation oracle at 2e-5/2e-4).
[[nodiscard]] auto qwen_tiled_exp(Var<float> x) noexcept {
    return exp2(x * 1.4426950408889634f);
}
}// namespace

[[nodiscard]] uint32_t qwen35_attn_prefill_tiled_shared_bytes(const QwenAttnPrefillTiledConfig &cfg) noexcept {
    // One K tile + one V tile, one contiguous row per key: the cooperative load
    // is coalesced and the lane-strided reads are bank-conflict-free.
    return 2u * cfg.k_tile * cfg.head_dim * static_cast<uint32_t>(sizeof(float));
}

[[nodiscard]] Status validate_qwen35_attn_prefill_tiled_config(const QwenAttnPrefillTiledConfig &cfg) noexcept {
    if (cfg.head_dim == 0u || cfg.head_dim % 32u != 0u) {
        return err_invalid("tiled attention head_dim must be a positive multiple of 32");
    }
    if (cfg.q_heads == 0u || cfg.kv_heads == 0u || cfg.q_heads % cfg.kv_heads != 0u) {
        return err_invalid("tiled attention GQA requires q_heads % kv_heads == 0");
    }
    if (cfg.q_heads / cfg.kv_heads != 4u) {
        return err_invalid("tiled attention requires exactly 4 q heads per kv head (one warp each)");
    }
    if (cfg.q_tile == 0u || cfg.q_tile * (cfg.head_dim / 32u) > 64u) {
        return err_invalid("tiled attention q_tile * (head_dim/32) must be in [1,64] (register accumulator)");
    }
    if (cfg.k_tile == 0u || cfg.k_tile > 64u || (cfg.k_tile * cfg.head_dim) % 128u != 0u) {
        return err_invalid("tiled attention k_tile * head_dim must tile the 128-thread block load");
    }
    if (qwen35_attn_prefill_tiled_shared_bytes(cfg) == 0u ||
        qwen35_attn_prefill_tiled_shared_bytes(cfg) < 2u * cfg.head_dim * static_cast<uint32_t>(sizeof(float))) {
        return err_invalid("tiled attention shared-memory footprint is degenerate");
    }
    return Status::success();
}

[[nodiscard]] Status select_qwen35_attn_prefill_tiled_config(uint32_t warp_size,
                                                             uint32_t max_shared_memory_bytes,
                                                             QwenAttnPrefillTiledConfig &cfg) noexcept {
    if (warp_size != 32u) {
        return Status::error(StatusCode::unsupported,
                             "tiled prefill attention pins warp size 32",
                             luisa::string{"device reports warp size "} + luisa::string{std::to_string(warp_size).c_str()});
    }
    QwenAttnPrefillTiledConfig candidate{};
    if (auto s = validate_qwen35_attn_prefill_tiled_config(candidate); !s.ok()) { return s; }
    const auto floor_bytes = qwen35_attn_prefill_tiled_shared_bytes(candidate);
    // 0 means "this backend does not report a limit" (DX overrides nothing and
    // returns 0), which is NOT a device without shared memory: it selects the
    // portable 16 KiB tile that fits D3D12's 32 KiB per-group ceiling.
    if (max_shared_memory_bytes != 0u && max_shared_memory_bytes < floor_bytes) {
        return Status::error(StatusCode::unsupported,
                             "device shared memory is below the tiled attention tile floor",
                             luisa::string{"reports "} + luisa::string{std::to_string(max_shared_memory_bytes).c_str()} +
                                 " B, needs " + luisa::string{std::to_string(floor_bytes).c_str()} + " B");
    }
    // Deliberately NO "use a bigger tile because the device can" upgrade: on the
    // target box the unrolled per-tile body reaches the 255-register ceiling at
    // larger extents, so occupancy (not shared memory) is the binding resource
    // and bigger tiles measured slower (q4/k16 3.49 ms vs q2/k2 2.32 ms at
    // S=1024). A caller wanting a different trade can pass an explicit config.
    cfg = candidate;
    return Status::success();
}

[[nodiscard]] QwenAttnPrefillTiledKernel make_qwen35_attn_prefill_tiled_kernel(
    const QwenAttnPrefillTiledConfig &cfg, QwenKvStorage kv) noexcept {
    const uint32_t hd = cfg.head_dim;
    const uint32_t qh = cfg.q_heads;
    const uint32_t kvh = cfg.kv_heads;
    const uint32_t gqa = qh / kvh;// warps per block, one q head each
    const uint32_t q_tile = cfg.q_tile;
    const uint32_t k_tile = cfg.k_tile;
    const uint32_t dims = hd / 32u;         // per-lane channels of one row
    // KV cache storage (the dtype/layout branches fold away: compile-time).
    const bool kv_blocked = kv.blocked();
    const uint32_t q_stride = qh * hd; // q/gate/out row stride [q_heads, hd]
    const uint32_t load_iters = (k_tile * hd) / 128u;
    // Masked / floor scores use a finite sentinel, never +/-inf: the online
    // softmax must stay free of inf*0 and inf-inf on every backend lowering (the
    // Vulkan XIR->SPIR-V path mis-lowers -inf here), and the sentinel must stay
    // finite after the log2(e) scaling below. Any real QK score is |s| <= 1e3.
    constexpr float kScoreFloor = -1.0e30f;
    QwenAttnPrefillTiledKernel kernel = [=](BufferFloat q, BufferFloat gate, BufferFloat k_cache,
                                            BufferFloat v_cache, BufferFloat k_scale, BufferFloat v_scale,
                                            BufferFloat out,
                                            Var<uint> tokens, Var<float> scale, Var<uint> pos0,
                                            Var<uint> kv_len, Var<uint> capacity) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name("qwen_attn_prefill_tiled");
        // Uniform per-block allocation: declare the K/V tiles at kernel scope.
        Shared<float> k_tiles{k_tile * hd};
        Shared<float> v_tiles{k_tile * hd};
        const auto block = qwen_linear_id() / 128u;
        const auto row_tile = block / kvh;
        const auto kv_head = block % kvh;
        // Block-uniform guard: every thread of a participating block enters, so
        // the warp collectives and the barriers stay in lockstep.
        $if (row_tile * q_tile < tokens) {
            const auto lane = warp_lane_id();
            const auto q_head = kv_head * gqa + thread_x() / 32u;
            const auto row0 = row_tile * q_tile;
            const auto n_keys = min(kv_len, capacity);// defensive clamp of the cache bound
            // Register state: q_tile rows x this lane's channel slice.
            Local<float> qreg{q_tile * dims};
            Local<float> acc{q_tile * dims};
            Local<float> kreg{dims};
            Local<float> vreg{dims};
            Local<float> srow{q_tile * k_tile};// masked scores of the current tile
            Local<float> run_m{q_tile};
            Local<float> run_l{q_tile};
            Local<float> tile_m{q_tile};
            for (uint32_t r = 0u; r < q_tile; r++) {
                const auto row = row0 + r;
                $if (row < tokens) {
                    const auto base = row * q_stride + q_head * hd + lane;
                    for (uint32_t c = 0u; c < dims; c++) { qreg[r * dims + c] = q.read(base + 32u * c); }
                } $else {
                    for (uint32_t c = 0u; c < dims; c++) { qreg[r * dims + c] = 0.0f; }
                };
                for (uint32_t c = 0u; c < dims; c++) { acc[r * dims + c] = 0.0f; }
                run_m[r] = kScoreFloor;
                run_l[r] = 0.0f;
            }
            // Causal window actually needed by this tile (ragged tail included).
            const auto row_last = min(row0 + q_tile - 1u, tokens - 1u);
            const auto key_end = min(pos0 + row_last + 1u, n_keys);
            $for (k0, 0u, key_end, k_tile) {
                // Cooperative coalesced staging of the K/V tile. Keys outside the
                // causal window are zero-filled so no lane ever consumes stale
                // shared memory (the masks below drop them anyway).
                for (uint32_t i = 0u; i < load_iters; i++) {
                    const auto lin = thread_x() + i * 128u;
                    const auto key = k0 + lin / hd;

                        const auto src = qwen_kv_code_index(key, kv_head, lin % hd, kvh, hd,
                                                            capacity, kv_blocked);
                        $if (key < key_end) {
                            // Stage the raw CODE: the block scale of a row is applied in the
                            // consumer loops below, where it is loaded once per key per lane
                            // instead of once per staged element.
                            k_tiles.write(lin, qwen_kv_code_value(k_cache, src, kv));
                            v_tiles.write(lin, qwen_kv_code_value(v_cache, src, kv));
                        } $else {
                            k_tiles.write(lin, 0.0f);
                            v_tiles.write(lin, 0.0f);
                        };
                }
                sync_block();
                // Scores: warp all-reduce over the channel axis, then the causal
                // mask (query row r sees keys [0, pos0 + row]) and the tile max.
                for (uint32_t r = 0u; r < q_tile; r++) { tile_m[r] = kScoreFloor; }
                for (uint32_t j = 0u; j < k_tile; j++) {
                    const auto key = k0 + j;
                    const auto in_range = key < key_end;
                    // One shared read per (key, lane channel): the K row of this
                    // key is reused by every row of the tile.
                    // One uniform scale load per key for the whole channel sweep
                    // (the compile-time 1 for an unscaled dtype folds away).
                    const auto kscale_j = qwen_kv_scale_at(k_scale, kv_head, key, capacity, kv);
                    for (uint32_t c = 0u; c < dims; c++) {
                        kreg[c] = k_tiles.read(j * hd + lane + 32u * c) * kscale_j;
                    }
                    for (uint32_t r = 0u; r < q_tile; r++) {
                        Var<float> dot = def(0.0f);
                        for (uint32_t c = 0u; c < dims; c++) {
                            dot = fma(qreg[r * dims + c], kreg[c], dot);
                        }
                        const auto score = qwen_warp_butterfly_sum(dot) * scale;
                        const auto row = row0 + r;
                        const auto visible = in_range & (key <= pos0 + row) & (row < tokens);
                        const auto masked = ite(visible, score, kScoreFloor);
                        srow[r * k_tile + j] = masked;
                        tile_m[r] = max(tile_m[r], masked);
                    }
                }
                // ONE accumulator rescale per row per tile, then P.V fused into
                // the same accumulators (one FMA per channel per key).
                for (uint32_t r = 0u; r < q_tile; r++) {
                    const auto m_new = max(run_m[r], tile_m[r]);
                    const auto corr = qwen_tiled_exp(run_m[r] - m_new);
                    run_m[r] = m_new;
                    run_l[r] = run_l[r] * corr;
                    for (uint32_t c = 0u; c < dims; c++) { acc[r * dims + c] = acc[r * dims + c] * corr; }
                }
                for (uint32_t j = 0u; j < k_tile; j++) {
                    // One uniform scale load per key for the whole channel sweep
                    // (the compile-time 1 for an unscaled dtype folds away).
                    const auto vscale_j = qwen_kv_scale_at(v_scale, kv_head, k0 + j, capacity, kv);
                    for (uint32_t c = 0u; c < dims; c++) {
                        vreg[c] = v_tiles.read(j * hd + lane + 32u * c) * vscale_j;
                    }
                    for (uint32_t r = 0u; r < q_tile; r++) {
                        // A key outside this row's causal window contributes
                        // exactly zero weight (and zero output), so a tile that is
                        // entirely masked leaves run_m/run_l/acc untouched.
                        const auto visible = srow[r * k_tile + j] > kScoreFloor;
                        const auto p = ite(visible, qwen_tiled_exp(srow[r * k_tile + j] - run_m[r]), 0.0f);
                        run_l[r] = run_l[r] + p;
                        for (uint32_t c = 0u; c < dims; c++) {
                            acc[r * dims + c] = fma(p, vreg[c], acc[r * dims + c]);
                        }
                    }
                }
                sync_block();
            };
            // Epilogue: normalise by the softmax denominator and apply the same
            // sigmoid output gate as the reference kernel.
            for (uint32_t r = 0u; r < q_tile; r++) {
                const auto row = row0 + r;
                $if (row < tokens) {
                    const auto inv_l = 1.0f / run_l[r];
                    const auto base = row * q_stride + q_head * hd + lane;
                    for (uint32_t c = 0u; c < dims; c++) {
                        const auto idx = base + 32u * c;
                        out.write(idx, acc[r * dims + c] * inv_l * qwen_sigmoid(gate.read(idx)));
                    }
                };
            }
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_attn_prefill_tiled_dispatch(
    uint64_t tokens, const QwenAttnPrefillTiledConfig &cfg) {
    if (auto s = validate_qwen35_attn_prefill_tiled_config(cfg); !s.ok()) {
        return Result<QwenLaunch1D>::failure(s);
    }
    QwenLaunch1D plan;
    if (tokens == 0u) { return Result<QwenLaunch1D>::success(plan); }
    // One 128-thread block per (query tile, kv head).
    const uint64_t tiles = (tokens + cfg.q_tile - 1u) / cfg.q_tile;
    const auto blocks = checked_product_u32(tiles, cfg.kv_heads, "tiled attention blocks");
    if (!blocks.ok()) { return Result<QwenLaunch1D>::failure(blocks.status); }
    const auto threads = checked_product_u32(*blocks.value, 128u, "tiled attention threads");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto planned = plan_qwen_threads(*threads.value, 128u);
    if (!planned.ok()) { return Result<QwenLaunch1D>::failure(planned.status); }
    planned.value->work_items = *blocks.value;
    planned.value->warp_size = 32u;
    return planned;
}

[[nodiscard]] Status validate_qwen35_attn_prefill_tiled_inputs(uint64_t tokens, uint64_t pos0,
                                                               uint64_t kv_len, uint64_t capacity,
                                                               const QwenAttnPrefillTiledConfig &cfg) {
    if (auto s = validate_qwen35_attn_prefill_tiled_config(cfg); !s.ok()) { return s; }
    // Identical causal/capacity contract as the reference kernel.
    return validate_qwen_attn_prefill_inputs(tokens, cfg.q_heads, cfg.kv_heads, cfg.head_dim,
                                             pos0, kv_len, capacity);
}

namespace {

// Shared body of the block-cooperative GQA decode attention. `kv_len` is a host
// scalar in the reference realization and state[qwen_dec_state_kv_len] in the
// device-state one; everything else is identical. The shared merge buffers are
// declared by the caller (kernel scope, uniform allocation) and passed in.
  void qwen_attn_decode_body(const BufferFloat &q, const BufferFloat &gate, const BufferFloat &k_cache, const BufferFloat &v_cache,
                               const BufferFloat &k_scale, const BufferFloat &v_scale,
                               const BufferFloat &out, Var<uint> kv_len,
                               Shared<float> &sm, Shared<float> &sl, Shared<float> &sacc,
                               Var<uint> q_heads, Var<uint> kv_heads,
                               Var<uint> head_dim, Var<float> scale,
                               Var<uint> capacity, uint32_t kv_len_limit,
                               QwenKvStorage kv) noexcept {
      set_block_size(128u, 1u, 1u);
      set_warp_size(32u);
      // Block index from the linear thread id (works for flattened plans).
      const auto h = qwen_linear_id() / 128u;
      // Block-uniform guard: all 128 threads of a participating block enter.
      // `kv_len_limit` is a compile-time host constant (0 = unbounded): a nonzero
      // limit makes the whole block return immediately once the split-KV family
      // owns the range, which keeps one captured CUDA graph valid across the
      // threshold without any per-token update (opt_plan/03 section 4.2).
      auto participates = h < q_heads & head_dim > 0u & head_dim <= 256u & kv_heads > 0u & kv_len > 0u;
      if (kv_len_limit != 0u) { participates = participates & kv_len < kv_len_limit; }
      $if (participates) {
        const auto lane = warp_lane_id();
        const auto wid = thread_x() / 32u;// warp index in block (0..3)
        const auto kvh = h / (q_heads / kv_heads);
        const auto q_base = h * head_dim;
        const auto nkeys = min(kv_len, capacity);// defensive clamp
        // Per-warp online softmax over the strided key subset
        // key = wid, wid+4, wid+8, ... Lane l owns dims l, l+32, ...
        // A warp with an EMPTY key subset keeps m=-inf sentinel, l=0,
        // acc=0; the merge below skips it (exp(-inf - m_max) = 0).
        Var<float> run_m = def(-3.0e38f);
        Var<float> run_l = def(0.0f);
        Local<float> acc{256u};// decode validator: head_dim <= 256
        $for (d, lane, head_dim, 32u) { acc[d] = 0.0f; };
        $for (key, wid, nkeys, 4u) {

                Var<float> partial = def(0.0f);
                $for (d, lane, head_dim, 32u) {
                    partial += q.read(q_base + d) * qwen_kv_load_at(k_cache, k_scale, key, kvh, d, kv_heads,
                                                                    head_dim, capacity, kv);
            };
            const auto score = warp_active_sum(partial) * scale;
            const auto new_m = max(run_m, score);
            const auto corr = exp(run_m - new_m);
            const auto p = exp(score - new_m);
            run_l = run_l * corr + p;
            $for (d, lane, head_dim, 32u) {
                    acc[d] = acc[d] * corr +
                             p * qwen_kv_load_at(v_cache, v_scale, key, kvh, d, kv_heads, head_dim,
                                                 capacity, kv);
            };
            run_m = new_m;
        };
        // Publish per-warp partials: m/l in shared scalars, acc in a
        // shared [4, head_dim] tile (buffers declared at kernel scope).
        $if (lane == 0u) {
            sm[wid] = run_m;
            sl[wid] = run_l;
        };
        $for (d, lane, head_dim, 32u) {
            sacc[wid * 256u + d] = acc[d];
        };
        sync_block();
        // Merge in warp order, but distribute independent output channels over
        // all 128 threads. The baseline gave thread 0 a serial 256-channel
        // loop and recomputed the same four exponentials for every channel.
        Var<float> m_max = def(-3.0e38f);
        $for (w, 4u) {
            $if (sl.read(w) > 0.0f) { m_max = max(m_max, sm.read(w)); };
        };
        Var<float> l_tot = def(0.0f);
        $for (w, 4u) {
            $if (sl.read(w) > 0.0f) { l_tot += sl.read(w) * exp(sm.read(w) - m_max); };
        };
        const auto inv_l = 1.0f / l_tot;
        Local<float> corr{4u};
        $for (w, 4u) {
            corr[w] = 0.0f;
            $if (sl.read(w) > 0.0f) { corr[w] = exp(sm.read(w) - m_max); };
        };
        $for (d, thread_x(), head_dim, 128u) {
            Var<float> o = def(0.0f);
            $for (w, 4u) {
                o += sacc.read(w * 256u + d) * corr[w];
            };
            out.write(q_base + d, o * inv_l * qwen_sigmoid(gate.read(q_base + d)));
        };
    };
}

  }// namespace
  [[nodiscard]] QwenAttnDecodeKernel make_qwen_attn_decode_kernel(uint32_t kv_len_limit,
                                                       QwenKvStorage kv) noexcept {
    QwenAttnDecodeKernel kernel = [kv_len_limit, kv](BufferFloat q, BufferFloat gate, BufferFloat k_cache, BufferFloat v_cache,
                                                   BufferFloat k_scale, BufferFloat v_scale,
                                                   BufferFloat out, Var<uint> q_heads, Var<uint> kv_heads,
                                                   Var<uint> head_dim, Var<float> scale, Var<uint> kv_len,
                                                   Var<uint> capacity) noexcept {
          set_name("qwen_attn_decode");
          // Shared merge buffers at kernel scope (uniform allocation).
          Shared<float> sm{4u};
          Shared<float> sl{4u};
          Shared<float> sacc{4u * 256u};// head_dim <= 256 for decode (validator)

            qwen_attn_decode_body(q, gate, k_cache, v_cache, k_scale, v_scale, out, kv_len,
                                  sm, sl, sacc, q_heads, kv_heads, head_dim, scale, capacity,
                                  kv_len_limit, kv);
      };
      return kernel;
  }
  [[nodiscard]] QwenAttnDecodeDevKernel make_qwen_attn_decode_dev_kernel(uint32_t kv_len_limit,
                                                          QwenKvStorage kv) noexcept {
    QwenAttnDecodeDevKernel kernel = [kv_len_limit, kv](BufferFloat q, BufferFloat gate, BufferFloat k_cache, BufferFloat v_cache,
                                                      BufferFloat k_scale, BufferFloat v_scale,
                                                      BufferFloat out, BufferUInt state, Var<uint> q_heads, Var<uint> kv_heads,
                                                      Var<uint> head_dim, Var<float> scale,
                                                      Var<uint> capacity) noexcept {
          set_name("qwen_attn_decode_dev");
          Shared<float> sm{4u};
          Shared<float> sl{4u};
          Shared<float> sacc{4u * 256u};
          // The step's prepare appended its own K/V at state[position], so the
          // readable prefix is state[kv_len] (== position + 1 by contract).

            qwen_attn_decode_body(q, gate, k_cache, v_cache, k_scale, v_scale, out,
                                  state.read(qwen_dec_state_kv_len),
                                  sm, sl, sacc, q_heads, kv_heads, head_dim, scale, capacity,
                                  kv_len_limit, kv);
      };
      return kernel;
  }

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_decode_dispatch(uint64_t q_heads) {
    // One 128-thread block per q_head.
    const auto threads = checked_product_u32(q_heads, 128u, "decode threads");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto plan = plan_qwen_threads(*threads.value, 128u);
    if (!plan.ok()) { return Result<QwenLaunch1D>::failure(plan.status); }
    plan.value->work_items = q_heads;
    plan.value->warp_size = 32u;
    return plan;
}

[[nodiscard]] Status validate_qwen_attn_decode_inputs(uint64_t q_heads, uint64_t kv_heads,
                                                         uint64_t head_dim, uint64_t kv_len, uint64_t capacity) {
    if (q_heads == 0u || kv_heads == 0u || q_heads % kv_heads != 0u) {
        return err_invalid("decode GQA requires q_heads % kv_heads == 0");
    }
    if (head_dim == 0u || head_dim % 32u != 0u || head_dim > 256u) {
        return err_invalid("decode head_dim must be a multiple of 32 in [32,256] (shared tile bound)");
    }
    if (kv_len == 0u) { return err_invalid("decode requires kv_len >= 1 (append the current token first)"); }
    if (kv_len > capacity) { return err_invalid("decode kv_len exceeds cache capacity"); }
    const auto cache_n = checked_product_u32(capacity, kv_heads, "decode cache");
    const auto q_n = checked_product_u32(q_heads, head_dim, "decode queries");
      if (!cache_n.ok() || !q_n.ok() ||
          !checked_product_u32(*cache_n.value, head_dim, "decode cache").ok()) {
          return err_overflow("decode element count exceeds uint32");
      }
      return Status::success();
  }

  // ---------------------------------------------------------------------------
  // Split-KV decode attention with GQA dedup (opt_plan/03 section 4.2).
  // See dsl_ops.h for the contract; the two bodies below are the whole kernel.
  // ---------------------------------------------------------------------------
  [[nodiscard]] uint32_t qwen_attn_decode_split_shared_bytes(const QwenAttnDecodeSplitConfig &cfg) noexcept {
      // One K tile + one V tile, one contiguous [k_tile, head_dim] block each, so
      // the 128-thread cooperative load is coalesced and the lane-strided reads
      // (lane + 32*c) are bank-conflict-free.
      return 2u * cfg.k_tile * cfg.head_dim * static_cast<uint32_t>(sizeof(float));
  }

  [[nodiscard]] uint64_t qwen_attn_decode_split_partial_bytes(const QwenAttnDecodeSplitConfig &cfg) noexcept {
      const uint64_t acc = static_cast<uint64_t>(cfg.max_splits) * cfg.q_heads * cfg.head_dim;
      const uint64_t ml = static_cast<uint64_t>(cfg.max_splits) * cfg.q_heads * 2u;
      return (acc + ml) * sizeof(float);
  }

  [[nodiscard]] Status validate_qwen_attn_decode_split_config(const QwenAttnDecodeSplitConfig &cfg) noexcept {
      if (cfg.head_dim == 0u || cfg.head_dim % 32u != 0u || cfg.head_dim > 256u) {
          return err_invalid("split decode head_dim must be a multiple of 32 in [32,256]");
      }
      if (cfg.kv_heads == 0u || cfg.q_heads == 0u || cfg.q_heads % cfg.kv_heads != 0u) {
          return err_invalid("split decode GQA requires q_heads % kv_heads == 0");
      }
      if (cfg.q_heads / cfg.kv_heads != 4u) {
          return err_invalid("split decode requires exactly 4 q heads per kv head (one warp each)");
      }
      if (cfg.k_tile == 0u || cfg.k_tile > 64u || (cfg.k_tile * cfg.head_dim) % 128u != 0u) {
          return err_invalid("split decode k_tile * head_dim must tile the 128-thread block load");
      }
      if (qwen_attn_decode_split_shared_bytes(cfg) == 0u ||
          qwen_attn_decode_split_shared_bytes(cfg) <
              2u * cfg.head_dim * static_cast<uint32_t>(sizeof(float))) {
          return err_invalid("split decode shared-memory footprint is degenerate");
      }
      if (cfg.split_chunk == 0u) {
          return err_invalid("split decode split_chunk must be >= 1");
      }
      if (cfg.max_splits == 0u || cfg.max_splits > qwen_attn_decode_split_cap) {
          return err_invalid("split decode max_splits must be in [1, qwen_attn_decode_split_cap]");
      }
      if (cfg.threshold == 0u) {
          return err_invalid("split decode threshold must be >= 1 (the legacy kernel owns kv_len 0)");
      }
      // Fixed partial buffers: every index the kernels form is a uint32 product
      // of compile-time values, so the bound is checked once here (overflow-safe).
      const auto acc = checked_product_u32(cfg.max_splits, cfg.q_heads, "split partial acc");
      if (!acc.ok() || !checked_product_u32(*acc.value, cfg.head_dim, "split partial acc").ok()) {
          return err_overflow("split decode partial accumulator exceeds uint32");
      }
      const auto acc_elems = static_cast<uint64_t>(cfg.max_splits) * cfg.q_heads * cfg.head_dim;
      if (acc_elems > 0xFFFFFFFFull) {
          return err_overflow("split decode partial accumulator exceeds uint32");
      }
      const auto blocks = checked_product_u32(cfg.kv_heads, cfg.max_splits, "split decode blocks");
      if (!blocks.ok() || !checked_product_u32(*blocks.value, 128u, "split decode threads").ok()) {
          return err_overflow("split decode launch exceeds uint32");
      }
      return Status::success();
  }

  [[nodiscard]] uint32_t qwen_attn_decode_split_bound(uint32_t sm_count, uint32_t kv_heads) noexcept {
      if (kv_heads == 0u) { return 1u; }
      // No SM count was measured: fail closed to the documented default instead
      // of guessing a device property (see dsl_ops.h and qwen/caps.h).
      if (sm_count == 0u) { return qwen_attn_decode_split_default_bound; }
      // ~2 blocks/SM, the measured knee (see dsl_ops.h). uint64 math: a caller is
      // free to pass any sm_count, and the product must not wrap before the clamp.
      const auto by_sm = static_cast<uint64_t>(qwen_attn_decode_split_blocks_per_sm) *
                         sm_count / kv_heads;
      return static_cast<uint32_t>(
          std::max<uint64_t>(1u, std::min<uint64_t>(by_sm, qwen_attn_decode_split_cap)));
  }

  [[nodiscard]] Status select_qwen_attn_decode_split_config(uint32_t warp_size,
                                                            uint32_t max_shared_memory_bytes,
                                                            uint32_t sm_count,
                                                            QwenAttnDecodeSplitConfig &cfg) noexcept {
      if (warp_size != 32u) {
          return Status::error(StatusCode::unsupported,
                               "split decode attention pins warp size 32",
                               luisa::string{"device reports warp size "} +
                                   luisa::string{std::to_string(warp_size).c_str()});
      }
      QwenAttnDecodeSplitConfig candidate{};
      // Static shared-memory FLOOR: one key per tile is the smallest realizable
      // staging buffer, i.e. K tile + V tile = 2 * head_dim floats (2 KiB at the
      // model's head_dim 256). A device that reports less than that cannot run this
      // kernel at all, so the selection fails closed instead of handing the caller an
      // image the driver would reject.
      //
      // The pinned CUDA backend reports
      // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK (48 KiB on Ada/consumer
      // Blackwell), and this DSL kernel declares a STATIC/uniform `Shared`
      // allocation, which cannot use the opt-in dynamic carveout at all: the reported
      // number IS the budget.
      const auto floor_bytes = static_cast<uint32_t>(2u * candidate.head_dim * sizeof(float));
      if (max_shared_memory_bytes != 0u && max_shared_memory_bytes < floor_bytes) {
          return Status::error(StatusCode::unsupported,
                               "device shared memory is below the split decode tile floor",
                               luisa::string{"reports "} +
                                   luisa::string{std::to_string(max_shared_memory_bytes).c_str()} +
                                   " B, needs " + luisa::string{std::to_string(floor_bytes).c_str()} + " B");
      }
      // k_tile: the MEASURED optimum (8 keys = 16 KiB) whenever the reported limit
      // allows it, else the largest power of two below it that fits
      // (2 * k_tile * head_dim * 4 B <= limit). See dsl_ops.h for why the measured
      // tile is the small one (occupancy, not staging, is the binding resource) and
      // why the limit is an upper bound the chosen tile must NEVER exceed: a device
      // reporting 4/8/12 KiB gets 2/4/4 keys instead of a 16 KiB image the driver
      // rejects, which turns a hard compile failure into a smaller tile.
      // A 0-byte report (DX: no limit exposed) keeps the portable 16 KiB tile, and so
      // does every backend at or above 16 KiB (48 KiB on the target box).
      uint32_t k_tile = 8u;
      if (max_shared_memory_bytes != 0u) {
          while (k_tile > 1u &&
                 2ull * k_tile * candidate.head_dim * sizeof(float) > max_shared_memory_bytes) {
              k_tile >>= 1u;
          }
      }
      candidate.k_tile = k_tile;
      candidate.max_splits = qwen_attn_decode_split_bound(sm_count, candidate.kv_heads);
      if (auto s = validate_qwen_attn_decode_split_config(candidate); !s.ok()) { return s; }
      // Post-condition (fail closed): the tile handed to the caller fits the reported
      // limit. Unreachable through the down-clamp above, but never worth assuming when
      // the alternative is a kernel compile failure on the device.
      const auto shared_bytes = qwen_attn_decode_split_shared_bytes(candidate);
      if (max_shared_memory_bytes != 0u && shared_bytes > max_shared_memory_bytes) {
          return Status::error(StatusCode::unsupported,
                               "split decode K/V tile exceeds the reported shared-memory limit",
                               luisa::string{"tile needs "} +
                                   luisa::string{std::to_string(shared_bytes).c_str()} +
                                   " B, device reports " +
                                   luisa::string{std::to_string(max_shared_memory_bytes).c_str()} + " B");
      }
      cfg = candidate;
      return Status::success();
  }

  [[nodiscard]] Status qwen_attn_decode_split_apply_max_splits_override(
      QwenAttnDecodeSplitConfig &cfg, uint32_t max_splits,
      uint32_t max_shared_memory_bytes) noexcept {
      // 0 keeps the bound the selection derived from the device.
      if (max_splits == 0u) { return Status::success(); }
      if (max_splits > qwen_attn_decode_split_cap) {
          return err_invalid(luisa::string{"split decode max_splits override "} +
                             luisa::string{std::to_string(max_splits).c_str()} +
                             " is above the plan bound " +
                             luisa::string{std::to_string(qwen_attn_decode_split_cap).c_str()});
      }
      auto candidate = cfg;
      candidate.max_splits = max_splits;
      // The override only moves the compile-time grid bound, so the kernel contract
      // (and the fixed partial workspace it indexes) must be re-validated: that is the
      // uint32 overflow gate. `cfg` is left untouched on refusal, so a rejected
      // override can never leave a half-applied config behind.
      if (auto s = validate_qwen_attn_decode_split_config(candidate); !s.ok()) { return s; }
      // The K/V tile is the one the shared-memory limit selected, and raising the split
      // bound must not spend shared memory: re-assert the budget so a hand-built config
      // cannot slip an over-budget tile through this path either.
      const auto shared_bytes = qwen_attn_decode_split_shared_bytes(candidate);
      if (max_shared_memory_bytes != 0u && shared_bytes > max_shared_memory_bytes) {
          return Status::error(StatusCode::unsupported,
                               "split decode K/V tile exceeds the reported shared-memory limit",
                               luisa::string{"tile needs "} +
                                   luisa::string{std::to_string(shared_bytes).c_str()} +
                                   " B, device reports " +
                                   luisa::string{std::to_string(max_shared_memory_bytes).c_str()} + " B");
      }
      cfg = candidate;
      return Status::success();
  }

  [[nodiscard]] uint32_t qwen_attn_decode_split_count(uint64_t kv_len,
                                                      const QwenAttnDecodeSplitConfig &cfg) noexcept {
      if (kv_len == 0u) { return 0u; }
      const auto chunks = (kv_len + cfg.split_chunk - 1u) / cfg.split_chunk;
      const auto capped = std::min<uint64_t>(chunks, cfg.max_splits);
      return static_cast<uint32_t>(std::max<uint64_t>(capped, 1u));
  }

  [[nodiscard]] uint32_t qwen_attn_decode_split_span(uint64_t kv_len, uint32_t split,
                                                     const QwenAttnDecodeSplitConfig &cfg) noexcept {
      const auto splits = qwen_attn_decode_split_count(kv_len, cfg);
      if (splits == 0u || split >= splits) { return 0u; }
      const auto per = (kv_len + splits - 1u) / splits;
      const auto begin = static_cast<uint64_t>(split) * per;
      if (begin >= kv_len) { return 0u; }
      return static_cast<uint32_t>(std::min(kv_len, begin + per) - begin);
  }

  [[nodiscard]] uint32_t qwen_attn_decode_split_blocks(const QwenAttnDecodeSplitConfig &cfg) noexcept {
      return cfg.kv_heads * cfg.max_splits;
  }

  namespace {

  // Sentinel for "no key / empty split". Finite (never -inf) for the same reason
  // as the tiled prefill kernel: the Vulkan XIR->SPIR-V lowering mis-handles -inf
  // through the online-softmax rescaling, and the combine kernel drops empty
  // splits by their l <= 0 instead of by their m.
  constexpr float qwen_split_score_floor = -1.0e30f;

  // Shared body of the split-KV decode attention. `kv_len` is a host scalar in
  // the reference realization and state[qwen_dec_state_kv_len] in the
  // device-state one. The shared K/V tiles are declared by the caller (kernel
  // scope, uniform allocation) and passed in.
  void qwen_attn_decode_split_body(const BufferFloat &q, const BufferFloat &k_cache,
                                     const BufferFloat &v_cache, const BufferFloat &k_scale,
                                     const BufferFloat &v_scale, const BufferFloat &partial_acc,
                                     const BufferFloat &partial_ml, Var<uint> kv_len,
                                     Var<float> scale, Var<uint> capacity,
                                     const QwenAttnDecodeSplitConfig &cfg,
                                     Shared<float> &k_tiles, Shared<float> &v_tiles,
                                     QwenKvStorage kv) noexcept {
      const uint32_t hd = cfg.head_dim;
      const uint32_t qh = cfg.q_heads;
      const uint32_t kvh = cfg.kv_heads;
      const uint32_t gqa = qh / kvh;
      const uint32_t k_tile = cfg.k_tile;
      const uint32_t dims = hd / 32u;
      const bool kv_blocked = kv.blocked();
      const uint32_t load_iters = (k_tile * hd) / 128u;
      const uint32_t max_splits = cfg.max_splits;
      const uint32_t split_chunk = cfg.split_chunk;
      const uint32_t threshold = cfg.threshold;
      set_block_size(128u, 1u, 1u);
      set_warp_size(32u);
      const auto block = qwen_linear_id() / 128u;
      const auto kv_head = block / max_splits;
      const auto split = block % max_splits;
      // Block-uniform guard: the whole block enters or none of it does, so the
      // warp collectives and the tile barriers stay in lockstep. The legacy
      // per-q_head kernel owns kv_len < threshold (compile-time constant).
      const auto n_keys = min(kv_len, capacity);// defensive clamp of the cache bound
      $if (n_keys >= threshold) {
          const auto lane = warp_lane_id();
          const auto warp = thread_x() / 32u;
          const auto q_head = kv_head * gqa + warp;
          // Split count from the SAME kv_len the legacy kernel reads: the grid is
          // fixed at kvh * max_splits for every token, so a captured graph never
          // needs a new launch geometry (opt_plan/03 section 2.3 + 4.2).
          const auto splits = min(max((n_keys + split_chunk - 1u) / split_chunk, 1u), max_splits);
          const auto per = (n_keys + splits - 1u) / splits;
          const auto key_begin = split * per;
          const auto key_end = min(n_keys, key_begin + per);
          const auto acc_base = (split * qh + q_head) * hd;
          // FP32 online-softmax state of THIS warp's q head over its split range.
          // Every accumulator is a compile-time-sized Local (dims = head_dim/32),
          // so nothing is indexed by a runtime head_dim.
          Local<float> acc{dims};
          Local<float> qreg{dims};
          Local<float> kreg{dims};
          Local<float> vreg{dims};
          for (uint32_t c = 0u; c < dims; c++) { acc[c] = 0.0f; }
          Var<float> run_m = def(qwen_split_score_floor);
          Var<float> run_l = def(0.0f);
          $if (key_begin < key_end) {
              const auto q_base = q_head * hd + lane;
              for (uint32_t c = 0u; c < dims; c++) { qreg[c] = q.read(q_base + 32u * c); }
              $for (k0, key_begin, key_end, k_tile) {
                  // Cooperative coalesced staging of one K tile and one V tile.
                  // Keys outside this split's range are zero-filled so no lane
                  // ever consumes stale shared memory (the mask below drops them).
                  for (uint32_t i = 0u; i < load_iters; i++) {
                      const auto lin = thread_x() + i * 128u;
                      const auto key = k0 + lin / hd;

                        const auto src = qwen_kv_code_index(key, kv_head, lin % hd, kvh, hd,
                                                            capacity, kv_blocked);
                        $if (key < key_end) {
                            // Stage the raw CODE: the block scale of a row is applied in the
                            // consumer loops below, where it is loaded once per key per lane
                            // instead of once per staged element.
                            k_tiles.write(lin, qwen_kv_code_value(k_cache, src, kv));
                            v_tiles.write(lin, qwen_kv_code_value(v_cache, src, kv));
                        } $else {
                            k_tiles.write(lin, 0.0f);
                            v_tiles.write(lin, 0.0f);
                        };
                  }
                  sync_block();
                  // Per-warp online softmax. Each of the 4 warps of the block reads
                  // the SAME staged K/V tile for a different q head: that is the
                  // 4x GQA dedup (one global fetch per cache row per token).
                  for (uint32_t j = 0u; j < k_tile; j++) {
                      const auto visible = k0 + j < key_end;
                      // One uniform scale load per key for the whole channel sweep
                      // (the compile-time 1 for an unscaled dtype folds away).
                      const auto kscale_j = qwen_kv_scale_at(k_scale, kv_head, k0 + j, capacity, kv);
                      for (uint32_t c = 0u; c < dims; c++) {
                          kreg[c] = k_tiles.read(j * hd + lane + 32u * c) * kscale_j;
                      }
                      Var<float> dot = def(0.0f);
                      for (uint32_t c = 0u; c < dims; c++) {
                          dot = fma(qreg[c], kreg[c], dot);
                      }
                      const auto score = qwen_warp_butterfly_sum(dot) * scale;
                      const auto new_m = max(run_m, ite(visible, score, qwen_split_score_floor));
                      const auto corr = qwen_tiled_exp(run_m - new_m);
                      run_l = run_l * corr;
                      for (uint32_t c = 0u; c < dims; c++) { acc[c] = acc[c] * corr; }
                      // A key outside the split range contributes exactly zero
                      // weight (and zero output): a fully masked tile leaves the
                      // running statistics untouched.
                      const auto p = ite(visible, qwen_tiled_exp(score - new_m), 0.0f);
                      run_l = run_l + p;
                      // One uniform scale load per key for the whole channel sweep
                      // (the compile-time 1 for an unscaled dtype folds away).
                      const auto vscale_j = qwen_kv_scale_at(v_scale, kv_head, k0 + j, capacity, kv);
                      for (uint32_t c = 0u; c < dims; c++) {
                          vreg[c] = v_tiles.read(j * hd + lane + 32u * c) * vscale_j;
                      }
                      for (uint32_t c = 0u; c < dims; c++) {
                          acc[c] = fma(p, vreg[c], acc[c]);
                      }
                      run_m = new_m;
                  }
                  sync_block();
              };
          };
          // Publish this (kv_head, split, q_head) partial. EMPTY splits write
          // (floor, 0, 0), which the combine kernel skips by l <= 0, so a partial
          // of an earlier longer step is never read back.
          for (uint32_t c = 0u; c < dims; c++) {
              partial_acc.write(acc_base + lane + 32u * c, acc[c]);
          }
          $if (lane == 0u) {
              const auto ml = (split * qh + q_head) * 2u;
              partial_ml.write(ml, run_m);
              partial_ml.write(ml + 1u, run_l);
          };
      };
  }

  // Shared body of the combine kernel: reduce `splits` partials with the
  // rescaling identity, normalize by the softmax denominator and apply the
  // sigmoid gate. One 128-thread block per q_head; every thread walks the split
  // axis in ascending order, so the reduction order is fixed and deterministic.
  void qwen_attn_decode_combine_body(const BufferFloat &partial_acc, const BufferFloat &partial_ml,
                                     const BufferFloat &gate, const BufferFloat &out,
                                     Var<uint> kv_len, Var<uint> capacity,
                                     const QwenAttnDecodeSplitConfig &cfg) noexcept {
      const uint32_t hd = cfg.head_dim;
      const uint32_t qh = cfg.q_heads;
      const uint32_t max_splits = cfg.max_splits;
      const uint32_t split_chunk = cfg.split_chunk;
      const uint32_t threshold = cfg.threshold;
      const uint32_t dim_iters = (hd + 127u) / 128u;
      set_block_size(128u, 1u, 1u);
      set_warp_size(32u);
      const auto q_head = qwen_linear_id() / 128u;
      const auto n_keys = min(kv_len, capacity);
      $if (q_head < qh & n_keys >= threshold) {
          const auto splits = min(max((n_keys + split_chunk - 1u) / split_chunk, 1u), max_splits);
          // Running maximum over the non-empty splits.
          Var<float> m_max = def(qwen_split_score_floor);
          for (uint32_t s = 0u; s < max_splits; s++) {
              $if (s < splits) {
                  const auto ml = (s * qh + q_head) * 2u;
                  const auto l = partial_ml.read(ml + 1u);
                  $if (l > 0.0f) { m_max = max(m_max, partial_ml.read(ml)); };
              };
          }
          // Rescale weights of the non-empty splits + the total denominator.
          Local<float> corr{max_splits};
          Var<float> l_tot = def(0.0f);
          for (uint32_t s = 0u; s < max_splits; s++) {
              corr[s] = 0.0f;
              $if (s < splits) {
                  const auto ml = (s * qh + q_head) * 2u;
                  const auto l = partial_ml.read(ml + 1u);
                  $if (l > 0.0f) {
                      const auto w = qwen_tiled_exp(partial_ml.read(ml) - m_max);
                      corr[s] = w;
                      l_tot = l_tot + l * w;
                  };
              };
          }
          // A degenerate all-empty reduction writes a finite 0 instead of 0/0.
          const auto inv_l = ite(l_tot > 0.0f, 1.0f / l_tot, 0.0f);
          const auto base = q_head * hd;
          for (uint32_t i = 0u; i < dim_iters; i++) {
              const auto d = i * 128u + thread_x();
              $if (d < hd) {
                  Var<float> o = def(0.0f);
                  for (uint32_t s = 0u; s < max_splits; s++) {
                      o = fma(partial_acc.read((s * qh + q_head) * hd + d), corr[s], o);
                  }
                  out.write(base + d, o * inv_l * qwen_sigmoid(gate.read(base + d)));
              };
          }
      };
  }

  }// namespace

  [[nodiscard]] QwenAttnDecodeSplitKernel make_qwen_attn_decode_split_kernel(
    const QwenAttnDecodeSplitConfig &cfg, QwenKvStorage kv) noexcept {
      const uint32_t tile_floats = cfg.k_tile * cfg.head_dim;
      QwenAttnDecodeSplitKernel kernel = [cfg, tile_floats, kv](
                                             BufferFloat q, BufferFloat k_cache, BufferFloat v_cache,
                                             BufferFloat k_scale, BufferFloat v_scale,
                                             BufferFloat partial_acc, BufferFloat partial_ml,
                                             Var<uint> kv_len, Var<float> scale,
                                             Var<uint> capacity) noexcept {
          set_name("qwen_attn_decode_split");
          Shared<float> k_tiles{tile_floats};
          Shared<float> v_tiles{tile_floats};

            qwen_attn_decode_split_body(q, k_cache, v_cache, k_scale, v_scale, partial_acc, partial_ml,
                                        kv_len, scale, capacity, cfg, k_tiles, v_tiles, kv);
      };
      return kernel;
  }

  [[nodiscard]] QwenAttnDecodeSplitDevKernel make_qwen_attn_decode_split_dev_kernel(
    const QwenAttnDecodeSplitConfig &cfg, QwenKvStorage kv) noexcept {
      const uint32_t tile_floats = cfg.k_tile * cfg.head_dim;
      QwenAttnDecodeSplitDevKernel kernel = [cfg, tile_floats, kv](
                                                BufferFloat q, BufferFloat k_cache, BufferFloat v_cache,
                                                BufferFloat k_scale, BufferFloat v_scale,
                                                BufferFloat partial_acc, BufferFloat partial_ml,
                                                BufferUInt state, Var<float> scale,
                                                Var<uint> capacity) noexcept {
          set_name("qwen_attn_decode_split_dev");
          Shared<float> k_tiles{tile_floats};
          Shared<float> v_tiles{tile_floats};

            qwen_attn_decode_split_body(q, k_cache, v_cache, k_scale, v_scale, partial_acc, partial_ml,
                                        state.read(qwen_dec_state_kv_len), scale, capacity,
                                        cfg, k_tiles, v_tiles, kv);
      };
      return kernel;
  }

  [[nodiscard]] QwenAttnDecodeCombineKernel make_qwen_attn_decode_combine_kernel(
      const QwenAttnDecodeSplitConfig &cfg) noexcept {
      QwenAttnDecodeCombineKernel kernel = [cfg](BufferFloat partial_acc, BufferFloat partial_ml,
                                                 BufferFloat gate, BufferFloat out,
                                                 Var<uint> kv_len,
                                                 Var<uint> capacity) noexcept {
          set_name("qwen_attn_decode_combine");
          qwen_attn_decode_combine_body(partial_acc, partial_ml, gate, out, kv_len, capacity, cfg);
      };
      return kernel;
  }

  [[nodiscard]] QwenAttnDecodeCombineDevKernel make_qwen_attn_decode_combine_dev_kernel(
      const QwenAttnDecodeSplitConfig &cfg) noexcept {
      QwenAttnDecodeCombineDevKernel kernel = [cfg](BufferFloat partial_acc, BufferFloat partial_ml,
                                                    BufferFloat gate, BufferFloat out,
                                                    BufferUInt state,
                                                    Var<uint> capacity) noexcept {
          set_name("qwen_attn_decode_combine_dev");
          qwen_attn_decode_combine_body(partial_acc, partial_ml, gate, out,
                                        state.read(qwen_dec_state_kv_len), capacity, cfg);
      };
      return kernel;
  }

  [[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_decode_split_dispatch(
      const QwenAttnDecodeSplitConfig &cfg) {
      if (auto s = validate_qwen_attn_decode_split_config(cfg); !s.ok()) {
          return Result<QwenLaunch1D>::failure(s);
      }
      // STATIC grid: kv_heads * max_splits blocks of 128 threads for EVERY kv_len.
      const auto blocks = cfg.kv_heads * cfg.max_splits;
      const auto threads = checked_product_u32(blocks, 128u, "split decode threads");
      if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
      auto plan = plan_qwen_threads(*threads.value, 128u);
      if (!plan.ok()) { return Result<QwenLaunch1D>::failure(plan.status); }
      plan.value->work_items = blocks;
      plan.value->warp_size = 32u;
      return plan;
  }

  [[nodiscard]] Result<QwenLaunch1D> plan_qwen_attn_decode_combine_dispatch(
      const QwenAttnDecodeSplitConfig &cfg) {
      if (auto s = validate_qwen_attn_decode_split_config(cfg); !s.ok()) {
          return Result<QwenLaunch1D>::failure(s);
      }
      const auto threads = checked_product_u32(cfg.q_heads, 128u, "split decode combine threads");
      if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
      auto plan = plan_qwen_threads(*threads.value, 128u);
      if (!plan.ok()) { return Result<QwenLaunch1D>::failure(plan.status); }
      plan.value->work_items = cfg.q_heads;
      plan.value->warp_size = 32u;
      return plan;
  }

  [[nodiscard]] Status validate_qwen_attn_decode_split_inputs(uint64_t kv_len, uint64_t capacity,
                                                              const QwenAttnDecodeSplitConfig &cfg) {
      if (auto s = validate_qwen_attn_decode_split_config(cfg); !s.ok()) { return s; }
      if (kv_len == 0u) { return err_invalid("split decode requires kv_len >= 1"); }
      if (kv_len > capacity) { return err_invalid("split decode kv_len exceeds cache capacity"); }
      if (kv_len > 0xFFFFFFFFull) { return err_overflow("split decode kv_len exceeds uint32"); }
      const auto cache_n = checked_product_u32(capacity, cfg.kv_heads, "split decode cache");
      if (!cache_n.ok() || !checked_product_u32(*cache_n.value, cfg.head_dim, "split decode cache").ok()) {
          return err_overflow("split decode element count exceeds uint32");
      }
      return Status::success();
  }


// ---------------------------------------------------------------------------
// 8. Gated DeltaNet
// ---------------------------------------------------------------------------
namespace {

// Shared body of the depthwise conv + SiLU with per-layer ring history. `base`
// is the absolute position of token 0: a host kernel argument in the reference
// realization and state[qwen_dec_state_position] in the device-state one, which
// is what keeps the 24 GDN layers of a static decode graph argument-invariant
// while the ring still advances one position per launch (opt_plan/09 V3).
void qwen_gdn_conv_silu_body(const BufferFloat &src, const BufferFloat &weight, const BufferFloat &ring, const BufferFloat &out,
                             Var<uint> base, Var<uint> tokens, Var<uint> channels,
                             Var<uint> kernel_size) noexcept {
    set_block_size(256u, 1u, 1u);
    const auto c = qwen_linear_id();
    $if (c < channels & kernel_size >= 2u & tokens > 0u) {
        const auto rw = kernel_size - 1u;
        // Causal order inside the thread: a ring slot is overwritten
        // only AFTER every output that reads the older tap was computed.
        $for (t, tokens) {
            const auto p = base + t;
            Var<float> acc = def(0.0f);
            $for (j, kernel_size) {
                const auto signed_pos = cast<int>(p) - cast<int>(rw) + cast<int>(j);
                Var<float> xv = def(0.0f);
                $if (signed_pos >= 0) {
                    const auto abs_pos = cast<uint>(signed_pos);
                    $if (abs_pos < base) {
                        xv = ring.read(c * rw + (abs_pos % rw));
                    } $else {
                        xv = src.read((abs_pos - base) * channels + c);
                    };
                };
                // GGUF ssm_conv1d [kernel, channels] payload is CHANNEL-major:
                // weight[c*kernel + j] is tap j of channel c.
                acc += weight.read(c * kernel_size + j) * xv;
            };
            out.write(t * channels + c, qwen_silu(acc));
            ring.write(c * rw + (p % rw), src.read(t * channels + c));
        };
    };
}

}// namespace

[[nodiscard]] QwenGdnConvKernel make_qwen_gdn_conv_silu_kernel() noexcept {
    QwenGdnConvKernel kernel = [](BufferFloat src, BufferFloat weight, BufferFloat ring, BufferFloat out,
                                  Var<uint> tokens, Var<uint> channels, Var<uint> kernel_size,
                                  Var<uint> pos0) noexcept {
        set_name("qwen_gdn_conv_silu");
        qwen_gdn_conv_silu_body(src, weight, ring, out, pos0, tokens, channels, kernel_size);
    };
    return kernel;
}

[[nodiscard]] QwenGdnConvDevKernel make_qwen_gdn_conv_silu_dev_kernel() noexcept {
    QwenGdnConvDevKernel kernel = [](BufferFloat src, BufferFloat weight, BufferFloat ring, BufferFloat out,
                                     BufferUInt state, Var<uint> tokens, Var<uint> channels,
                                     Var<uint> kernel_size) noexcept {
        set_name("qwen_gdn_conv_silu_dev");
        qwen_gdn_conv_silu_body(src, weight, ring, out, state.read(qwen_dec_state_position),
                                tokens, channels, kernel_size);
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_conv_dispatch(uint64_t channels) {
    return plan_qwen_threads(channels);
}

[[nodiscard]] Status validate_qwen_gdn_conv_inputs(uint64_t tokens, uint64_t channels,
                                                      uint64_t kernel_size, uint64_t pos0) {
    if (channels == 0u || kernel_size < 2u) {
        return err_invalid("GDN conv geometry must be nonzero (kernel >= 2)");
    }
    if (!checked_product_u32(tokens, channels, "GDN conv src").ok() ||
        !checked_product_u32(channels, kernel_size, "GDN conv weight").ok() ||
        !checked_product_u32(channels, kernel_size - 1u, "GDN conv ring").ok()) {
        return err_overflow("GDN conv element count exceeds uint32");
    }
    if (pos0 > 0x7FFFFFFFull || tokens > 0x7FFFFFFFull - pos0) {
        return err_overflow("GDN conv absolute position exceeds int32 (signed tap math)");
    }
    return Status::success();
}

[[nodiscard]] QwenGdnGatesKernel make_qwen_gdn_gates_kernel() noexcept {
    QwenGdnGatesKernel kernel = [](BufferFloat a, BufferFloat b, BufferFloat ssm_a, BufferFloat dt_bias,
                                   BufferFloat beta_out, BufferFloat g_out,
                                   Var<uint> tokens, Var<uint> heads, Var<float> softplus_threshold) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("qwen_gdn_gates");
        const auto id = qwen_linear_id();
        $if (id < tokens * heads & heads > 0u) {
            const auto h = id % heads;
            const auto av = a.read(id) + dt_bias.read(h);
            Var<float> sp = def(0.0f);
            $if (av > softplus_threshold) {
                sp = av;
            } $else {
                sp = log(1.0f + exp(av));
            };
            beta_out.write(id, qwen_sigmoid(b.read(id)));
            // GGUF ssm_a is ALREADY -exp(A_log) (negative, folded offline by
            // the converter): g = ssm_a[h] * softplus(a + dt_bias[h]).
            g_out.write(id, ssm_a.read(h) * sp);
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_gates_dispatch(uint64_t tokens, uint64_t heads) {
    const auto n = checked_product_u32(tokens, heads, "gdn gates");
    if (!n.ok()) { return Result<QwenLaunch1D>::failure(n.status); }
    return plan_qwen_threads(*n.value);
}

[[nodiscard]] QwenGdnDecodeGatesKernel make_qwen_gdn_decode_gates_kernel(
    QwenWeightFormat format) noexcept {
    QwenGdnDecodeGatesKernel kernel = [format](
        BufferFloat x, BufferUInt alpha_w, BufferFloat alpha_scales,
        BufferUInt beta_w, BufferFloat beta_scales, BufferFloat ssm_a,
        BufferFloat dt_bias, BufferFloat beta_out, BufferFloat g_out,
        Var<uint> k, Var<uint> heads, Var<uint> group,
        Var<float> softplus_threshold) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(luisa::string{"qwen_gdn_decode_gates"} + qwen_format_name_suffix(format));
        Shared<float> lut{format == QwenWeightFormat::fp8_e4m3 ? 256u : 1u};
        if (format == QwenWeightFormat::fp8_e4m3) {
            $for (i, thread_x(), 256u, 128u) { lut[i] = qwen_fp8_e4m3_to_f32(i); };
            sync_block();
        }
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < heads * 2u & heads > 0u & k > 0u) {
            const auto is_beta = warp >= heads;
            Var<uint> h = warp;
            $if (is_beta) { h = warp - heads; };
            const auto dot = [&](const BufferVar<uint32_t> &w, const BufferVar<float> &scales,
                                 Var<uint> row) {
                const auto w_base = row * k;
                Var<uint> scale_base = def(0u);
                if (format != QwenWeightFormat::bf16) {
                    scale_base = row * ((k + group - 1u) / group);
                }
                return qwen35_decode_dot(x, w, scales, lut, w_base, scale_base,
                                         k, group, lane, format);
            };
            Var<float> acc = def(0.0f);
            $if (is_beta) {
                acc = dot(beta_w, beta_scales, h);
            } $else {
                acc = dot(alpha_w, alpha_scales, h);
            };
            acc = warp_active_sum(acc);
            $if (lane == 0u) {
                $if (is_beta) {
                    beta_out.write(h, qwen_sigmoid(acc));
                } $else {
                    const auto av = acc + dt_bias.read(h);
                    Var<float> sp = def(0.0f);
                    $if (av > softplus_threshold) {
                        sp = av;
                    } $else {
                        sp = log(1.0f + exp(av));
                    };
                    g_out.write(h, ssm_a.read(h) * sp);
                };
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_decode_gates_dispatch(
    uint64_t heads, uint64_t k, QwenWeightFormat format, uint32_t group) {
    const auto rows = checked_product_u32(heads, 2u, "gdn decode gate rows");
    if (!rows.ok()) { return Result<QwenLaunch1D>::failure(rows.status); }
    if (k == 0u || (format == QwenWeightFormat::bf16 && k % 256u != 0u) ||
        (format == QwenWeightFormat::fp8_e4m3 &&
         (group == 0u || group % 4u != 0u || k % 512u != 0u))) {
        return Result<QwenLaunch1D>::failure(err_invalid("gdn decode gate packed geometry mismatch"));
    }
    if (format == QwenWeightFormat::w4) {
        if (auto st = validate_qwen_w4_geometry(k, group); !st.ok()) {
            return Result<QwenLaunch1D>::failure(std::move(st));
        }
    }
    if (format == QwenWeightFormat::q8_0) {
        if (auto st = validate_qwen_q8_0_geometry(k, group); !st.ok()) {
            return Result<QwenLaunch1D>::failure(std::move(st));
        }
    }
    const auto wn = checked_product_u32(heads, k, "gdn decode gate weights");
    if (!wn.ok()) { return Result<QwenLaunch1D>::failure(wn.status); }
    return plan_warp_rows(*rows.value, 128u);
}

// 128-bit twin of the T=1 alpha/beta gate fusion (opt_plan/09 N2). Same
// (2*heads)-warp geometry and the same gate epilogues; 128-bit weight loads, one
// FP8 scale per uint4, float4 activations and the branchless LUT-free decode.
[[nodiscard]] QwenGdnDecodeGatesPackedKernel make_qwen_gdn_decode_gates_packed_kernel(
    QwenWeightFormat format) noexcept {
    const bool fp8 = format == QwenWeightFormat::fp8_e4m3;
    const bool q8 = format == QwenWeightFormat::q8_0;
    const bool scaled = fp8 || q8;
    QwenGdnDecodeGatesPackedKernel kernel = [fp8, q8, scaled, format](
        BufferFloat4 x, BufferUInt4 alpha_w, BufferFloat alpha_scales,
        BufferUInt4 beta_w, BufferFloat beta_scales, BufferFloat ssm_a,
        BufferFloat dt_bias, BufferFloat beta_out, BufferFloat g_out,
        Var<uint> k, Var<uint> heads, Var<uint> group,
        Var<float> softplus_threshold, Var<uint> vecs) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name(luisa::string{"qwen_gdn_decode_gates_packed"} + qwen_format_name_suffix(format));
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < heads * 2u & heads > 0u & k > 0u) {
            const auto is_beta = warp >= heads;
            Var<uint> h = warp;
            $if (is_beta) { h = warp - heads; };
            const auto w_base = h * vecs;
            Var<uint> scale_base = def(0u);
            if (scaled) { scale_base = h * ((k + group - 1u) / group); }
            Var<uint> group_vecs = def(1u);
            if (scaled) { group_vecs = group >> 4u; }
            // One stream per warp (the row is already chosen), four accumulators.
            const auto dot = [&](const BufferVar<uint4> &w, const BufferVar<float> &sc) {
                Var<float> a0 = def(0.0f), a1 = def(0.0f), a2 = def(0.0f), a3 = def(0.0f);
                $for (vi, lane, vecs, 32u) {
                    const auto wv = w.read(w_base + vi);
                    if (scaled) {
                        const auto s = sc.read(scale_base + vi / group_vecs);
                        const auto v0 = x.read(vi * 4u);
                        const auto v1 = x.read(vi * 4u + 1u);
                        const auto v2 = x.read(vi * 4u + 2u);
                        const auto v3 = x.read(vi * 4u + 3u);
                        a0 += v0.x * qwen_scaled_byte_to_f32(wv.x & 0xFFu, format) * s;
                        a0 += v0.y * qwen_scaled_byte_to_f32((wv.x >> 8u) & 0xFFu, format) * s;
                        a0 += v0.z * qwen_scaled_byte_to_f32((wv.x >> 16u) & 0xFFu, format) * s;
                        a0 += v0.w * qwen_scaled_byte_to_f32((wv.x >> 24u) & 0xFFu, format) * s;
                        a1 += v1.x * qwen_scaled_byte_to_f32(wv.y & 0xFFu, format) * s;
                        a1 += v1.y * qwen_scaled_byte_to_f32((wv.y >> 8u) & 0xFFu, format) * s;
                        a1 += v1.z * qwen_scaled_byte_to_f32((wv.y >> 16u) & 0xFFu, format) * s;
                        a1 += v1.w * qwen_scaled_byte_to_f32((wv.y >> 24u) & 0xFFu, format) * s;
                        a2 += v2.x * qwen_scaled_byte_to_f32(wv.z & 0xFFu, format) * s;
                        a2 += v2.y * qwen_scaled_byte_to_f32((wv.z >> 8u) & 0xFFu, format) * s;
                        a2 += v2.z * qwen_scaled_byte_to_f32((wv.z >> 16u) & 0xFFu, format) * s;
                        a2 += v2.w * qwen_scaled_byte_to_f32((wv.z >> 24u) & 0xFFu, format) * s;
                        a3 += v3.x * qwen_scaled_byte_to_f32(wv.w & 0xFFu, format) * s;
                        a3 += v3.y * qwen_scaled_byte_to_f32((wv.w >> 8u) & 0xFFu, format) * s;
                        a3 += v3.z * qwen_scaled_byte_to_f32((wv.w >> 16u) & 0xFFu, format) * s;
                        a3 += v3.w * qwen_scaled_byte_to_f32((wv.w >> 24u) & 0xFFu, format) * s;
                    } else {
                        const auto v0 = x.read(vi * 2u);
                        const auto v1 = x.read(vi * 2u + 1u);
                        a0 += v0.x * qwen_bf16_to_f32(wv.x & 0xFFFFu);
                        a0 += v0.y * qwen_bf16_to_f32((wv.x >> 16u) & 0xFFFFu);
                        a1 += v0.z * qwen_bf16_to_f32(wv.y & 0xFFFFu);
                        a1 += v0.w * qwen_bf16_to_f32((wv.y >> 16u) & 0xFFFFu);
                        a2 += v1.x * qwen_bf16_to_f32(wv.z & 0xFFFFu);
                        a2 += v1.y * qwen_bf16_to_f32((wv.z >> 16u) & 0xFFFFu);
                        a3 += v1.z * qwen_bf16_to_f32(wv.w & 0xFFFFu);
                        a3 += v1.w * qwen_bf16_to_f32((wv.w >> 16u) & 0xFFFFu);
                    };
                };
                return (a0 + a1) + (a2 + a3);
            };
            Var<float> acc = def(0.0f);
            $if (is_beta) {
                acc = dot(beta_w, beta_scales);
            } $else {
                acc = dot(alpha_w, alpha_scales);
            };
            acc = warp_active_sum(acc);
            $if (lane == 0u) {
                $if (is_beta) {
                    beta_out.write(h, qwen_sigmoid(acc));
                } $else {
                    const auto av = acc + dt_bias.read(h);
                    Var<float> sp = def(0.0f);
                    $if (av > softplus_threshold) {
                        sp = av;
                    } $else {
                        sp = log(1.0f + exp(av));
                    };
                    g_out.write(h, ssm_a.read(h) * sp);
                };
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_decode_gates_packed_dispatch(
    uint64_t heads, uint64_t k, QwenWeightFormat format, uint32_t group) {
    const auto rows = checked_product_u32(heads, 2u, "gdn decode gate rows");
    if (!rows.ok()) { return Result<QwenLaunch1D>::failure(rows.status); }
    if (auto s = validate_packed_decode_geometry(k, format, group); !s.ok()) {
        return Result<QwenLaunch1D>::failure(std::move(s));
    }
    const auto wn = checked_product_u32(heads, k, "gdn decode gate weights");
    if (!wn.ok()) { return Result<QwenLaunch1D>::failure(wn.status); }
    return plan_warp_rows(*rows.value, 128u);
}

namespace {

// Shared body of the fused decode conv + SiLU + ring update + QK L2 norm
// (opt_plan/09 N3.3). `base` is the absolute position of the single token: a
// host argument in the reference realization, state[position] in the
// device-state one.
void qwen35_gdn_decode_fused_body(const BufferFloat &qkv, const BufferFloat &conv_w,
                                  const BufferFloat &ring, const BufferFloat &q_out,
                                  const BufferFloat &k_out, const BufferFloat &conv_out,
                                  Var<uint> base, Var<uint> qk_heads, Var<uint> head_dim,
                                  Var<uint> v_channels, Var<uint> kernel_size,
                                  Var<float> eps, Var<float> q_scale) noexcept {
    set_block_size(128u, 1u, 1u);
    set_warp_size(32u);
    // Model geometry: head_dim 128 -> four lane-owned channels per lane, one
    // warp per 128-channel head row.
    constexpr uint32_t kHD = qwen35_gdn_model_head_dim;
    constexpr uint32_t kLanes = kHD / 32u;
    const auto g = qwen_linear_id() / 32u;// global warp id == channel group
    const auto lane = warp_lane_id();
    const auto qk_n = qk_heads * head_dim;
    const auto channels = 2u * qk_n + v_channels;
    auto hd_ok = head_dim == static_cast<uint>(kHD);
    $if (g * kHD < channels & hd_ok & kernel_size >= 2u) {
        const auto rw = kernel_size - 1u;
        const auto c0 = g * kHD + lane;// channel base of this lane
        // Depthwise conv over the causal window, in ring order, exactly like the
        // unfused kernel: the ring slot is overwritten only after every tap that
        // reads history has been consumed for this channel.
        Local<float> conv{kLanes};
        for (uint32_t j = 0u; j < kLanes; j++) {
            const auto c = c0 + j * 32u;
            Var<float> acc = def(0.0f);
            $for (t, kernel_size) {
                const auto signed_pos = cast<int>(base) - cast<int>(rw) + cast<int>(t);
                Var<float> xv = def(0.0f);
                $if (signed_pos >= 0) {
                    const auto abs_pos = cast<uint>(signed_pos);
                    $if (abs_pos < base) {
                        xv = ring.read(c * rw + (abs_pos % rw));
                    } $else {
                        // base + 0 is this token: (abs_pos - base) == 0.
                        xv = qkv.read((abs_pos - base) * channels + c);
                    };
                };
                // GGUF ssm_conv1d [kernel, channels] payload is CHANNEL-major.
                acc += conv_w.read(c * kernel_size + t) * xv;
            };
            conv[j] = qwen_silu(acc);
            ring.write(c * rw + (base % rw), qkv.read(c));
        };
        // Q group: normalized and scaled, same reduction order as the unfused
        // qk-norm kernel (ascending lane elements, one warp reduction).
        $if (g < qk_heads) {
            Var<float> ss = def(0.0f);
            for (uint32_t j = 0u; j < kLanes; j++) { ss += conv[j] * conv[j]; };
            ss = warp_active_sum(ss);
            const auto inv = rsqrt(ss + eps);
            for (uint32_t j = 0u; j < kLanes; j++) {
                q_out.write(g * kHD + lane + j * 32u, conv[j] * inv * q_scale);
            };
        } $else {
            Var<bool> is_k = g < 2u * qk_heads;
            $if (is_k) {
                Var<float> ss = def(0.0f);
                for (uint32_t j = 0u; j < kLanes; j++) { ss += conv[j] * conv[j]; };
                ss = warp_active_sum(ss);
                const auto inv = rsqrt(ss + eps);
                const auto h = g - qk_heads;
                for (uint32_t j = 0u; j < kLanes; j++) {
                    k_out.write(h * kHD + lane + j * 32u, conv[j] * inv);
                };
            } $else {
                // V group: the recurrent kernel consumes the convolved values at
                // v_offset + h*kHD, i.e. exactly channel g*kHD + lane + j*32.
                for (uint32_t j = 0u; j < kLanes; j++) {
                    conv_out.write(c0 + j * 32u, conv[j]);
                };
            };
        };
    };
}

}// namespace

[[nodiscard]] QwenGdnDecodeFusedKernel make_qwen35_gdn_decode_fused_kernel() noexcept {
    QwenGdnDecodeFusedKernel kernel = [](BufferFloat qkv, BufferFloat conv_w, BufferFloat ring,
                                         BufferFloat q_out, BufferFloat k_out, BufferFloat conv_out,
                                         Var<uint> qk_heads, Var<uint> head_dim, Var<uint> v_channels,
                                         Var<uint> kernel_size, Var<float> eps, Var<float> q_scale,
                                         Var<uint> pos0) noexcept {
        set_name("qwen35_gdn_decode_fused");
        qwen35_gdn_decode_fused_body(qkv, conv_w, ring, q_out, k_out, conv_out, pos0,
                                     qk_heads, head_dim, v_channels, kernel_size, eps, q_scale);
    };
    return kernel;
}

[[nodiscard]] QwenGdnDecodeFusedDevKernel make_qwen35_gdn_decode_fused_dev_kernel() noexcept {
    QwenGdnDecodeFusedDevKernel kernel = [](BufferFloat qkv, BufferFloat conv_w, BufferFloat ring,
                                            BufferFloat q_out, BufferFloat k_out, BufferFloat conv_out,
                                            BufferUInt state, Var<uint> qk_heads, Var<uint> head_dim,
                                            Var<uint> v_channels, Var<uint> kernel_size,
                                            Var<float> eps, Var<float> q_scale) noexcept {
        set_name("qwen35_gdn_decode_fused_dev");
        qwen35_gdn_decode_fused_body(qkv, conv_w, ring, q_out, k_out, conv_out,
                                     state.read(qwen_dec_state_position),
                                     qk_heads, head_dim, v_channels, kernel_size, eps, q_scale);
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_gdn_decode_fused_dispatch(
    uint64_t qk_heads, uint64_t head_dim, uint64_t v_channels) {
    if (qk_heads == 0u || head_dim != qwen35_gdn_model_head_dim) {
        return Result<QwenLaunch1D>::failure(err_invalid(
            "fused GDN decode requires the model head_dim (128)"));
    }
    if (v_channels == 0u) {
        return Result<QwenLaunch1D>::failure(err_invalid("fused GDN decode needs a value channel count"));
    }
    const uint64_t channels = 2u * qk_heads * head_dim + v_channels;
    if (channels % qwen35_gdn_model_head_dim != 0u) {
        return Result<QwenLaunch1D>::failure(err_invalid(
            "fused GDN decode requires the channel count to be a multiple of the head width"));
    }
    const uint64_t warps = channels / qwen35_gdn_model_head_dim;
    const uint64_t blocks = (warps + 3u) / 4u;
    if (blocks == 0u || blocks > 0xFFFFFFFFull) {
        return Result<QwenLaunch1D>::failure(err_overflow("fused GDN decode grid exceeds uint32"));
    }
    const auto threads = checked_product_u32(blocks, 128u, "fused GDN decode threads");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto plan = plan_qwen_threads(*threads.value, 128u);
    if (!plan.ok()) { return plan; }
    plan.value->work_items = static_cast<uint32_t>(blocks);
    plan.value->warp_size = 32u;
    return plan;
}

[[nodiscard]] Status validate_qwen35_gdn_decode_fused_inputs(
    uint64_t tokens, uint64_t qk_heads, uint64_t head_dim, uint64_t v_channels,
    uint64_t kernel_size, uint64_t pos0) {
    if (tokens != 1u) {
        return Status::error(StatusCode::unsupported, "fused GDN decode is a single-token realization");
    }
    if (auto s = validate_qwen_gdn_conv_inputs(tokens, 2u * qk_heads * head_dim + v_channels,
                                               kernel_size, pos0);
        !s.ok()) {
        return s;
    }
    auto plan = plan_qwen35_gdn_decode_fused_dispatch(qk_heads, head_dim, v_channels);
    if (!plan.ok()) {
        return Status::error(StatusCode::invalid_argument,
                             "fused GDN decode geometry is not a model shape", plan.status.message);
    }
    return Status::success();
}

[[nodiscard]] QwenGdnQkNormKernel make_qwen_gdn_qk_l2norm_kernel() noexcept {
    QwenGdnQkNormKernel kernel = [](BufferFloat conv_out, BufferFloat q_out, BufferFloat k_out,
                                    Var<uint> tokens, Var<uint> qk_heads, Var<uint> head_dim,
                                    Var<uint> v_channels, Var<float> eps, Var<float> q_scale, Var<uint> rows) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name("qwen_gdn_qk_l2norm");
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        // rows = tokens*qk_heads; each warp does the Q row AND the K row of
        // one (token, qk_head). Block-uniform guard.
        $if (warp < rows & head_dim > 0u & tokens > 0u) {
            const auto t = warp / qk_heads;
            const auto h = warp % qk_heads;
            const auto stride = 2u * qk_heads * head_dim + v_channels;
            const auto q_src = t * stride + h * head_dim;
            const auto k_src = q_src + qk_heads * head_dim;
            const auto dst = (t * qk_heads + h) * head_dim;
            Var<float> q_ss = def(0.0f);
            Var<float> k_ss = def(0.0f);
            $for (i, lane, head_dim, 32u) {
                const auto qv = conv_out.read(q_src + i);
                const auto kv = conv_out.read(k_src + i);
                q_ss += qv * qv;
                k_ss += kv * kv;
            };
            q_ss = warp_active_sum(q_ss);
            k_ss = warp_active_sum(k_ss);
            // FLA/HF l2norm: x * rsqrt(sum(x*x) + eps) -- eps INSIDE the
            // rsqrt (near-zero rows stay near zero, never blown up to 1/eps).
            const auto q_inv = rsqrt(q_ss + eps);
            const auto k_inv = rsqrt(k_ss + eps);
            $for (i, lane, head_dim, 32u) {
                q_out.write(dst + i, conv_out.read(q_src + i) * q_inv * q_scale);
                k_out.write(dst + i, conv_out.read(k_src + i) * k_inv);
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_qk_norm_dispatch(uint64_t tokens, uint64_t qk_heads) {
    const auto rows = checked_product_u32(tokens, qk_heads, "gdn qk rows");
    if (!rows.ok()) { return Result<QwenLaunch1D>::failure(rows.status); }
    return plan_warp_rows(*rows.value, 128u);
}

namespace {

// Shared body of the per-token GDN recurrence. HD == 0 is the generic
// runtime-extent realization; HD == 128 is the Qwen3.5-4B model specialization
// (compile-time state-column extent: opt_plan/09 V4).
template<uint32_t HD>
void qwen_gdn_recurrent_body(const BufferFloat &q, const BufferFloat &k, const BufferFloat &conv_out,
                             const BufferFloat &beta, const BufferFloat &g, const BufferFloat &state,
                             const BufferFloat &out, Var<uint> tokens, Var<uint> v_heads,
                             Var<uint> head_dim, Var<uint> qk_heads, Var<uint> v_offset) noexcept {
    set_block_size(128u, 1u, 1u);
    set_warp_size(32u);
    // Shared staging buffers MUST be declared at kernel scope (not inside
    // a divergent $if) so every block allocates them uniformly.
    Shared<float> sk{HD == 0u ? 512u : HD};
    Shared<float> sq{HD == 0u ? 512u : HD};
    // Block index from the linear thread id (works for flattened plans).
    const auto h = qwen_linear_id() / 128u;
    auto hd_ok = head_dim > 0u;
    if constexpr (HD != 0u) { hd_ok = head_dim == static_cast<uint>(HD); }
    // Block-uniform guard. One lane owns one state column, so head_dim
    // must not exceed 128 lanes across the four 32-lane warps.
    $if (h < v_heads & hd_ok & head_dim <= 128u & (head_dim % 32u) == 0u & tokens > 0u) {
        const auto lane = warp_lane_id();
        const auto wid = thread_x() / 32u;
        const auto cols_per_warp = head_dim / 4u;
        // This thread owns value column v = wid*cols_per_warp + lane
        // (lanes >= cols_per_warp own nothing; all their reads/writes are
        // guarded and their local state column stays zero).
        const auto v_col = wid * cols_per_warp + lane;
        const auto owns_col = lane < cols_per_warp;
        // GGUF V-side tensors are TILED (converter reorders grouped HF
        // heads [K0_v0,K0_v1,K1_v0,...] -> [K0_v0,K1_v0,...,K0_v1,...]):
        // V head h pairs with QK head h % qk_heads (ggml tiled broadcast).
        const auto qkh = h % qk_heads;
        const auto s_base = h * head_dim * head_dim;
        const auto v_src_base = v_offset + h * head_dim;
        const auto c_stride = v_offset + v_heads * head_dim;
        // Thread-local state column S[:, v_col] (head_dim floats; fixed
        // 512-entry array, only [0,head_dim) used). Non-owning lanes keep
        // zeros so their reads never touch the state buffer.
        Local<float> col{HD == 0u ? 512u : HD};
        qwen_gdn_hd_loop<HD>(head_dim, [&](auto ki) { col[ki] = 0.0f; });
        $if (owns_col) {
            qwen_gdn_hd_loop<HD>(head_dim, [&](auto ki) {
                col[ki] = state.read(s_base + ki * head_dim + v_col);
            });
        };
        // Staged k/q vectors (shared, per token; declared at kernel scope).
        $for (t, tokens) {
            const auto th = t * v_heads + h;
            const auto alpha = exp(g.read(th));
            const auto bt = beta.read(th);
            const auto qk_row = (t * qk_heads + qkh) * head_dim;
            // Stage k and q for this token (all 128 threads cooperate).
            $for (i, thread_x(), head_dim, 128u) {
                sk[i] = k.read(qk_row + i);
                sq[i] = q.read(qk_row + i);
            };
            sync_block();
            // Decay the column and compute err_v = v[v] - sum_k S[k,v]
            // k[k]. Each lane owns a DIFFERENT value column, so the dot
            // is a per-lane serial reduction over the key axis (NOT a
            // warp collective -- that would mix columns). The k vector
            // is shared-broadcast, so all lanes read the same sk[ki].
            Var<float> dot = def(0.0f);
            qwen_gdn_hd_loop<HD>(head_dim, [&](auto ki) {
                col[ki] = col[ki] * alpha;
                dot += col[ki] * sk[ki];
            });
            // Guard the READ itself: non-owning lanes must not touch
            // conv_out (their v_col can exceed head_dim).
            Var<float> vv = def(0.0f);
            $if (owns_col) {
                vv = conv_out.read(t * c_stride + v_src_base + v_col);
            };
            const auto err = vv - dot;
            // Rank-1 update: S[k,v] += beta * k[k] * err.
            qwen_gdn_hd_loop<HD>(head_dim, [&](auto ki) {
                col[ki] = col[ki] + bt * sk[ki] * err;
            });
            // o[v] = sum_k S[k,v] q[k].
            Var<float> o = def(0.0f);
            qwen_gdn_hd_loop<HD>(head_dim, [&](auto ki) { o += col[ki] * sq[ki]; });
            $if (owns_col) {
                out.write((t * v_heads + h) * head_dim + v_col, o);
            };
            sync_block();
        };
        // Write the state column back once.
        $if (owns_col) {
            qwen_gdn_hd_loop<HD>(head_dim, [&](auto ki) {
                state.write(s_base + ki * head_dim + v_col, col[ki]);
            });
        };
    };
}

}// namespace

[[nodiscard]] QwenGdnRecurrentKernel make_qwen_gdn_recurrent_kernel() noexcept {
    QwenGdnRecurrentKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat conv_out,
                                       BufferFloat beta, BufferFloat g, BufferFloat state, BufferFloat out,
                                       Var<uint> tokens, Var<uint> v_heads, Var<uint> head_dim,
                                       Var<uint> qk_heads, Var<uint> v_offset) noexcept {
        set_name("qwen_gdn_recurrent");
        qwen_gdn_recurrent_body<0u>(q, k, conv_out, beta, g, state, out,
                                    tokens, v_heads, head_dim, qk_heads, v_offset);
    };
    return kernel;
}

[[nodiscard]] QwenGdnRecurrentKernel make_qwen35_gdn_recurrent_kernel(uint32_t head_dim) noexcept {
    if (head_dim != qwen35_gdn_model_head_dim) { return make_qwen_gdn_recurrent_kernel(); }
    QwenGdnRecurrentKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat conv_out,
                                       BufferFloat beta, BufferFloat g, BufferFloat state, BufferFloat out,
                                       Var<uint> tokens, Var<uint> v_heads, Var<uint> hd_arg,
                                       Var<uint> qk_heads, Var<uint> v_offset) noexcept {
        set_name("qwen35_gdn_recurrent");
        qwen_gdn_recurrent_body<qwen35_gdn_model_head_dim>(q, k, conv_out, beta, g, state, out,
                                                           tokens, v_heads, hd_arg, qk_heads, v_offset);
    };
    return kernel;
}

[[nodiscard]] QwenGdnRecurrentKernel make_qwen35_gdn_decode4_kernel() noexcept {
    QwenGdnRecurrentKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat conv_out,
                                       BufferFloat beta, BufferFloat g, BufferFloat state,
                                       BufferFloat out, Var<uint> tokens, Var<uint> v_heads,
                                       Var<uint> head_dim, Var<uint> qk_heads,
                                       Var<uint> v_offset) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name("qwen35_gdn_decode4");
        Shared<float> sk{128u};
        Shared<float> sq{128u};
        const auto work = qwen_linear_id() / 128u;
        $if (work < v_heads * 4u & tokens == 1u & head_dim == 128u) {
            const auto h = work / 4u;
            const auto col_group = work % 4u;
            const auto lane = warp_lane_id();
            const auto wid = thread_x() / 32u;
            const auto key_part = lane / 8u;
            const auto col_in_group = wid * 8u + (lane & 7u);
            const auto v_col = col_group * 32u + col_in_group;
            const auto key_begin = key_part * 32u;
            const auto qkh = h % qk_heads;
            const auto qk_row = qkh * 128u;
            $for (i, thread_x(), 128u, 128u) {
                sk[i] = k.read(qk_row + i);
                sq[i] = q.read(qk_row + i);
            };
            sync_block();
            const auto s_base = h * 128u * 128u + v_col;
            Local<float> col{32u};
            $for (j, 32u) {
                col[j] = state.read(s_base + (key_begin + j) * 128u);
            };
            const auto alpha = exp(g.read(h));
            const auto bt = beta.read(h);
            Var<float> dot = def(0.0f);
            $for (j, 32u) {
                col[j] = col[j] * alpha;
                dot += col[j] * sk[key_begin + j];
            };
            // Four key slices are lanes {c,c+8,c+16,c+24} in each warp.
            // Every lane participates in both shuffles and receives the sum.
            dot += warp_read_lane(dot, lane ^ 8u);
            dot += warp_read_lane(dot, lane ^ 16u);
            const auto vv = conv_out.read(v_offset + h * 128u + v_col);
            const auto err = vv - dot;
            Var<float> o = def(0.0f);
            $for (j, 32u) {
                col[j] = col[j] + bt * sk[key_begin + j] * err;
                o += col[j] * sq[key_begin + j];
            };
            o += warp_read_lane(o, lane ^ 8u);
            o += warp_read_lane(o, lane ^ 16u);
            $if (key_part == 0u) {
                out.write(h * 128u + v_col, o);
            };
            $for (j, 32u) {
                state.write(s_base + (key_begin + j) * 128u, col[j]);
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen35_gdn_decode4_dispatch(uint64_t v_heads) {
    const auto blocks = checked_product_u32(v_heads, 4u, "gdn decode4 blocks");
    if (!blocks.ok()) { return Result<QwenLaunch1D>::failure(blocks.status); }
    const auto threads = checked_product_u32(*blocks.value, 128u, "gdn decode4 threads");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto plan = plan_qwen_threads(*threads.value, 128u);
    if (!plan.ok()) { return plan; }
    plan.value->work_items = *blocks.value;
    plan.value->warp_size = 32u;
    return plan;
}

[[nodiscard]] Status validate_qwen35_gdn_decode4_inputs(
    uint64_t tokens, uint64_t v_heads, uint64_t qk_heads, uint64_t head_dim) {
    if (tokens != 1u || head_dim != 128u) {
        return err_invalid("qwen35 GDN decode4 requires T=1 and head_dim=128");
    }
    return validate_qwen_gdn_recurrent_inputs(tokens, v_heads, qk_heads, head_dim);
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_recurrent_dispatch(uint64_t v_heads) {
    const auto threads = checked_product_u32(v_heads, 128u, "gdn recurrent threads");
    if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
    auto plan = plan_qwen_threads(*threads.value, 128u);
    if (!plan.ok()) { return Result<QwenLaunch1D>::failure(plan.status); }
    plan.value->work_items = v_heads;
    plan.value->warp_size = 32u;
    return plan;
}

[[nodiscard]] Status validate_qwen_gdn_recurrent_inputs(uint64_t tokens, uint64_t v_heads,
                                                           uint64_t qk_heads, uint64_t head_dim) {
    if (v_heads == 0u || qk_heads == 0u || v_heads % qk_heads != 0u) {
        return err_invalid("GDN requires v_heads % qk_heads == 0");
    }
    if (head_dim == 0u || head_dim % 32u != 0u || head_dim > 128u) {
        return err_invalid("GDN head_dim must be a multiple of 32 in [32,128] (one state column per lane)");
    }
    if (tokens == 0u) { return Status::success(); }
    const auto v_n = checked_product_u32(tokens, v_heads, "GDN v rows");
    const auto s_n = checked_product_u32(v_heads, head_dim, "GDN state");
    const auto q_n = checked_product_u32(tokens, qk_heads, "GDN qk rows");
    if (!v_n.ok() || !s_n.ok() || !q_n.ok() ||
        !checked_product_u32(*v_n.value, head_dim, "GDN v elements").ok() ||
        !checked_product_u32(*s_n.value, head_dim, "GDN state elements").ok() ||
        !checked_product_u32(*q_n.value, head_dim, "GDN qk elements").ok()) {
        return err_overflow("GDN recurrent element count exceeds uint32");
    }
    return Status::success();
}

[[nodiscard]] uint64_t qwen_gdn_recurrent_shared_bytes(uint64_t head_dim) noexcept {
    // sk/sq are statically sized at 512 floats each (max head_dim).
    (void)head_dim;
    return 2u * 512u * sizeof(float);
}

[[nodiscard]] Status check_qwen_gdn_shared_budget(uint64_t head_dim, uint64_t backend_shared_bytes) {
    if (qwen_gdn_recurrent_shared_bytes(head_dim) > backend_shared_bytes) {
        return Status::error(StatusCode::unsupported,
                                "GDN recurrent kernel needs 4 KiB shared; backend limit is smaller");
    }
    return Status::success();
}

namespace {

// Head-dimension iteration for the GDN state-column kernels. With a compile-time
// extent every index is a constant, so a Local<> array indexed through `f` is
// register-resident (opt_plan/09 V4); the generic realization keeps the DSL
// device loop with a runtime Var<uint> index (the reference path).
template<uint32_t HD, typename F>
inline void qwen_gdn_hd_loop(Var<uint> head_dim, F &&f) noexcept {
    if constexpr (HD == 0u) {
        $for (i, head_dim) { f(i); };
    } else {
        for (uint32_t i = 0u; i < HD; i++) { f(i); };
    }
}

// Shared body of the WY/UT chunk-64 recurrence. HD == 0 is the generic
// runtime-extent realization; HD == 128 is the Qwen3.5-4B model specialization.
template<uint32_t HD>
void qwen_gdn_chunk_body(const BufferFloat &q, const BufferFloat &k, const BufferFloat &conv_out,
                         const BufferFloat &beta, const BufferFloat &g, const BufferFloat &state,
                         const BufferFloat &w, const BufferFloat &v_new, const BufferFloat &out,
                         Var<uint> tokens, Var<uint> v_heads, Var<uint> head_dim,
                         Var<uint> qk_heads, Var<uint> v_offset) noexcept {
    set_block_size(128u, 1u, 1u);
    // A holds the strict-lower KKT matrix, then its unit-lower UT inverse,
    // and finally the causal decay-scaled QK matrix. Reusing this 16 KiB
    // tile keeps the whole kernel below a 32 KiB shared-memory budget.
    Shared<float> a{qwen_gdn_chunk_size * qwen_gdn_chunk_size};
    Shared<float> sg{qwen_gdn_chunk_size};// chunk-local cumulative log decay
    Shared<float> sb{qwen_gdn_chunk_size};
    Shared<float> row{qwen_gdn_chunk_size};// UT forward-substitution staging
    const auto h = qwen_linear_id() / 128u;
    const auto lane = thread_x();
    // The specialized realization refuses any other head dimension instead of
    // indexing a 128-column state with the wrong stride.
    auto hd_ok = head_dim > 0u;
    if constexpr (HD != 0u) { hd_ok = head_dim == static_cast<uint>(HD); }
    $if (h < v_heads & hd_ok & head_dim <= 128u & qk_heads > 0u & tokens > 0u) {
        const auto qkh = h % qk_heads;
        const auto s_base = h * head_dim * head_dim;
        const auto v_base = v_offset + h * head_dim;
        const auto c_stride = v_offset + v_heads * head_dim;
        const auto owns_col = lane < head_dim;
        // The lane-owned state column is the chunk-start H_c. Keeping it
        // in registers avoids materializing NT copies of a 128x128 state.
        Local<float> col{HD == 0u ? 128u : HD};
        qwen_gdn_hd_loop<HD>(head_dim, [&](auto ki) { col[ki] = 0.0f; });
        $if (owns_col) {
            qwen_gdn_hd_loop<HD>(head_dim, [&](auto ki) {
                col[ki] = state.read(s_base + ki * head_dim + lane);
            });
        };
        $for (chunk, (tokens + qwen_gdn_chunk_size - 1u) / qwen_gdn_chunk_size) {
            const auto t0 = chunk * qwen_gdn_chunk_size;
            const auto count = min(qwen_gdn_chunk_size, tokens - t0);
            // Chunk-local FP32 cumsum. No global Gamma and no exp(-Gamma)
            // is formed, so long prompts cannot overflow transformed K.
            $if (lane == 0u) {
                Var<float> sum = def(0.0f);
                $for (i, count) {
                    const auto th = (t0 + i) * v_heads + h;
                    sum += g.read(th);
                    sg[i] = sum;
                    sb[i] = beta.read(th);
                };
            };
            sync_block();

            // A[i,j] = beta_i (k_i.k_j) exp(G_i-G_j), i>j.
            $for (idx, lane, qwen_gdn_chunk_size * qwen_gdn_chunk_size, 128u) {
                const auto i = idx / qwen_gdn_chunk_size;
                const auto j = idx % qwen_gdn_chunk_size;
                Var<float> value = def(0.0f);
                $if (i > j & i < count & j < count) {
                    const auto ki = ((t0 + i) * qk_heads + qkh) * head_dim;
                    const auto kj = ((t0 + j) * qk_heads + qkh) * head_dim;
                    Var<float> dot = def(0.0f);
                    qwen_gdn_hd_loop<HD>(head_dim, [&](auto d) {
                        dot += k.read(ki + d) * k.read(kj + d);
                    });
                    value = sb[i] * dot * exp(sg[i] - sg[j]);
                };
                a[idx] = value;
            };
            sync_block();

            // UT transform: A <- (I + strict_lower(A))^-1. Row i is
            // staged before any element overwrites the still-needed L_i,*
            // source row; preceding rows already contain inverse rows.
            $for (i, count) {
                $for (j, lane, i + 1u, 128u) {
                    Var<float> value = def(ite(i == j, 1.0f, 0.0f));
                    $for (l, j, i) { value -= a[i * qwen_gdn_chunk_size + l] *
                                              a[l * qwen_gdn_chunk_size + j]; };
                    row[j] = value;
                };
                sync_block();
                $for (j, lane, i + 1u, 128u) {
                    a[i * qwen_gdn_chunk_size + j] = row[j];
                };
                sync_block();
            };

            // W = A_inv * diag(beta * exp(G)) * K. Each chunk/head has
            // private scratch, so subsequent block-synchronized reads are
            // race-free without a second WY kernel launch.
            $for (idx, lane, count * head_dim, 128u) {
                const auto i = idx / head_dim;
                const auto d = idx % head_dim;
                Var<float> value = def(0.0f);
                $for (j, i + 1u) {
                    const auto kj = ((t0 + j) * qk_heads + qkh) * head_dim + d;
                    value += a[i * qwen_gdn_chunk_size + j] * sb[j] * exp(sg[j]) * k.read(kj);
                };
                w.write(((t0 + i) * v_heads + h) * head_dim + d, value);
            };
            sync_block();

            // v_new = U - W H_c, U=A_inv*diag(beta)*V. This remains
            // UNSCALED for the output equation; the state update below
            // applies exp(G_last-G_i) to its own copy of each value.
            $for (i, count) {
                $if (owns_col) {
                    Var<float> u = def(0.0f);
                    $for (j, i + 1u) {
                        const auto vv = conv_out.read((t0 + j) * c_stride + v_base + lane);
                        u += a[i * qwen_gdn_chunk_size + j] * sb[j] * vv;
                    };
                    Var<float> correction = def(0.0f);
                    qwen_gdn_hd_loop<HD>(head_dim, [&](auto d) {
                        correction += w.read(((t0 + i) * v_heads + h) * head_dim + d) * col[d];
                    });
                    v_new.write(((t0 + i) * v_heads + h) * head_dim + lane, u - correction);
                };
            };
            sync_block();

            // The UT inverse is dead now. Reuse A for the causal output
            // coefficient QK^T * exp(G_i-G_j), including the diagonal.
            $for (idx, lane, qwen_gdn_chunk_size * qwen_gdn_chunk_size, 128u) {
                const auto i = idx / qwen_gdn_chunk_size;
                const auto j = idx % qwen_gdn_chunk_size;
                Var<float> value = def(0.0f);
                $if (i >= j & i < count & j < count) {
                    const auto qi = ((t0 + i) * qk_heads + qkh) * head_dim;
                    const auto kj = ((t0 + j) * qk_heads + qkh) * head_dim;
                    Var<float> dot = def(0.0f);
                    qwen_gdn_hd_loop<HD>(head_dim, [&](auto d) {
                        dot += q.read(qi + d) * k.read(kj + d);
                    });
                    value = dot * exp(sg[i] - sg[j]);
                };
                a[idx] = value;
            };
            sync_block();

            // o_i = exp(G_i) q_i H_c + sum_{j<=i} A_out[i,j] v_new[j].
            $for (i, count) {
                $if (owns_col) {
                    const auto qi = ((t0 + i) * qk_heads + qkh) * head_dim;
                    Var<float> cross = def(0.0f);
                    qwen_gdn_hd_loop<HD>(head_dim, [&](auto d) {
                        cross += q.read(qi + d) * col[d];
                    });
                    cross *= exp(sg[i]);
                    Var<float> intra = def(0.0f);
                    $for (j, i + 1u) {
                        intra += a[i * qwen_gdn_chunk_size + j] *
                                 v_new.read(((t0 + j) * v_heads + h) * head_dim + lane);
                    };
                    out.write(((t0 + i) * v_heads + h) * head_dim + lane, cross + intra);
                };
            };

            // H_{c+1} = exp(G_last) H_c + K^T diag(exp(G_last-G_i)) v_new.
            // Each col[d] reads only its own previous value, so in-place
            // register updates cannot mix chunk-start and chunk-end states.
            const auto g_last = sg[count - 1u];
            $if (owns_col) {
                qwen_gdn_hd_loop<HD>(head_dim, [&](auto d) {
                    Var<float> value = def(col[d] * exp(g_last));
                    $for (i, count) {
                        const auto ki = ((t0 + i) * qk_heads + qkh) * head_dim + d;
                        const auto vn = v_new.read(((t0 + i) * v_heads + h) * head_dim + lane);
                        value += k.read(ki) * vn * exp(g_last - sg[i]);
                    };
                    col[d] = value;
                });
            };
            // No thread may overwrite A/W/v_new for the next chunk until
            // every output and state read from this chunk has completed.
            sync_block();
        };
        $if (owns_col) {
            qwen_gdn_hd_loop<HD>(head_dim, [&](auto d) {
                state.write(s_base + d * head_dim + lane, col[d]);
            });
        };
    };
}

}// namespace

[[nodiscard]] QwenGdnChunkKernel make_qwen_gdn_chunk_kernel() noexcept {
    QwenGdnChunkKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat conv_out,
                                   BufferFloat beta, BufferFloat g, BufferFloat state,
                                   BufferFloat w, BufferFloat v_new, BufferFloat out,
                                   Var<uint> tokens, Var<uint> v_heads, Var<uint> head_dim,
                                   Var<uint> qk_heads, Var<uint> v_offset) noexcept {
        set_name("qwen_gdn_wy_ut_chunk64");
        qwen_gdn_chunk_body<0u>(q, k, conv_out, beta, g, state, w, v_new, out,
                                tokens, v_heads, head_dim, qk_heads, v_offset);
    };
    return kernel;
}

[[nodiscard]] QwenGdnChunkKernel make_qwen35_gdn_chunk_kernel(uint32_t head_dim) noexcept {
    if (head_dim != qwen35_gdn_model_head_dim) { return make_qwen_gdn_chunk_kernel(); }
    QwenGdnChunkKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat conv_out,
                                   BufferFloat beta, BufferFloat g, BufferFloat state,
                                   BufferFloat w, BufferFloat v_new, BufferFloat out,
                                   Var<uint> tokens, Var<uint> v_heads, Var<uint> hd_arg,
                                   Var<uint> qk_heads, Var<uint> v_offset) noexcept {
        set_name("qwen35_gdn_wy_ut_chunk64");
        qwen_gdn_chunk_body<qwen35_gdn_model_head_dim>(q, k, conv_out, beta, g, state, w, v_new, out,
                                                       tokens, v_heads, hd_arg, qk_heads, v_offset);
    };
    return kernel;
}

[[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_chunk_dispatch(uint64_t v_heads) {
    return plan_qwen_gdn_recurrent_dispatch(v_heads);
}

[[nodiscard]] Status validate_qwen_gdn_chunk_inputs(uint64_t tokens, uint64_t v_heads,
                                                    uint64_t qk_heads, uint64_t head_dim) {
    if (auto s = validate_qwen_gdn_recurrent_inputs(tokens, v_heads, qk_heads, head_dim); !s.ok()) { return s; }
    const auto rows = checked_product_u32(tokens, v_heads, "GDN chunk scratch rows");
    if (!rows.ok() || !checked_product_u32(*rows.value, head_dim, "GDN chunk scratch").ok()) {
        return err_overflow("GDN chunk scratch element count exceeds uint32");
    }
    return Status::success();
}

[[nodiscard]] uint64_t qwen_gdn_chunk_shared_bytes() noexcept {
    return (qwen_gdn_chunk_size * qwen_gdn_chunk_size + 3u * qwen_gdn_chunk_size) * sizeof(float);
}

[[nodiscard]] Status dsl_ops_chunk_prefill_status() noexcept {
    return Status::success();
}

[[nodiscard]] QwenGdnGatedNormKernel make_qwen_gdn_gated_norm_kernel() noexcept {
    QwenGdnGatedNormKernel kernel = [](BufferFloat o, BufferFloat z, BufferFloat norm_w, BufferFloat out,
                                       Var<uint> tokens, Var<uint> heads, Var<uint> head_dim,
                                       Var<float> eps) noexcept {
        set_block_size(128u, 1u, 1u);
        set_warp_size(32u);
        set_name("qwen_gdn_gated_norm");
        const auto warp = qwen_linear_id() / 32u;
        const auto lane = warp_lane_id();
        $if (warp < tokens * heads & head_dim > 0u) {
            const auto base = warp * head_dim;
            Var<float> sumsq = def(0.0f);
            $for (i, lane, head_dim, 32u) {
                const auto v = o.read(base + i);
                sumsq += v * v;
            };
            sumsq = warp_active_sum(sumsq);
            const auto inv = rsqrt(sumsq / cast<float>(head_dim) + eps);
            // DIRECT weight (not 1+w); norm BEFORE the SiLU(z) gate.
            $for (i, lane, head_dim, 32u) {
                out.write(base + i, o.read(base + i) * inv * norm_w.read(i) * qwen_silu(z.read(base + i)));
            };
        };
    };
    return kernel;
}

  [[nodiscard]] Result<QwenLaunch1D> plan_qwen_gdn_gated_norm_dispatch(uint64_t tokens, uint64_t heads) {
      const auto rows = checked_product_u32(tokens, heads, "gdn gated norm rows");
      if (!rows.ok()) { return Result<QwenLaunch1D>::failure(rows.status); }
      return plan_warp_rows(*rows.value, 128u);
  }

  // ---------------------------------------------------------------------------
  // FUSED per-token GDN recurrence + gated output RMSNorm (opt_plan/03 section 5,
  // the one-CTA arm of the "GDN recurrent + gated norm" row).
  //
  // Same state update and recurrent output realization as
  // make_qwen35_gdn_recurrent_kernel(head_dim = 128) -- one 128-thread block per V
  // head, ONE state column per lane (v = thread_x()), the decayed column kept in a
  // register-resident Local<float>[128], the err/output dots as per-lane serial
  // reductions over the key axis with shared k/q broadcast, and the state written
  // back once after the (single) token -- and then, in the SAME block and without
  // any global round trip:
  //   1. the head's output row is staged in shared memory (`so`),
  //   2. sync_block(),
  //   3. the BLOCK-WIDE RMS reduction: every one of the four warps reads the staged
  //      row in the order and with the shuffle tree of
  //      make_qwen_gdn_gated_norm_kernel (i = lane, 128, 32 -> warp_active_sum), so
  //      all four warps derive a bit-identical sum with no cross-warp broadcast,
  //   4. each lane writes its own element of the DIRECT-weighted, SiLU(z)-gated row.
  // The default decode schedule cannot do this: make_qwen35_gdn_decode4_kernel owns
  // one quarter of the key axis per block, so its four blocks of a head cannot
  // reduce over the head's whole 128 channels -- hence the global _gdn_out round
  // trip plus a second launch there.
  // ---------------------------------------------------------------------------
  [[nodiscard]] QwenGdnRecurrentNormKernel make_qwen35_gdn_recurrent_norm_kernel() noexcept {
      QwenGdnRecurrentNormKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat conv_out,
                                             BufferFloat beta, BufferFloat g, BufferFloat state,
                                             BufferFloat z, BufferFloat norm_w, BufferFloat out,
                                             Var<uint> tokens, Var<uint> v_heads,
                                             Var<uint> head_dim, Var<uint> qk_heads,
                                             Var<uint> v_offset, Var<float> eps) noexcept {
          set_block_size(128u, 1u, 1u);
          set_warp_size(32u);
          set_name("qwen35_gdn_recurrent_norm");
          // Shared staging MUST be declared at kernel scope (not inside a divergent
          // $if) so every block allocates it uniformly.
          Shared<float> sk{qwen35_gdn_model_head_dim};
          Shared<float> sq{qwen35_gdn_model_head_dim};
          // Per-warp partial sums of the gated RMSNorm reduction. This is the whole
          // cross-warp state: the head row never leaves registers, so neither the
          // recurrence nor the norm touches global memory for it.
          Shared<float> sred{4u};
          const auto h = qwen_linear_id() / 128u;
          // Block-uniform guard: T=1 and the model head width only.
          $if (h < v_heads & tokens == 1u & head_dim == qwen35_gdn_model_head_dim) {
              const auto lane = warp_lane_id();
              // One state column per thread (128 threads, 128 columns): the compile-time
              // extent of the model specialization, so `col` stays register-resident.
              const auto v_col = thread_x();
              const auto qkh = h % qk_heads;
              const auto s_base = h * qwen35_gdn_model_head_dim * qwen35_gdn_model_head_dim;
              const auto v_src_base = v_offset + h * qwen35_gdn_model_head_dim;
              const auto row = h * qwen35_gdn_model_head_dim;
              // Thread-local state column S[:, v_col] (loaded once, updated in
              // registers, written back once).
              Local<float> col{qwen35_gdn_model_head_dim};
              for (uint32_t ki = 0u; ki < qwen35_gdn_model_head_dim; ki++) {
                  col[ki] = state.read(s_base + ki * qwen35_gdn_model_head_dim + v_col);
              }
              const auto alpha = exp(g.read(h));
              const auto bt = beta.read(h);
              const auto qk_row = qkh * qwen35_gdn_model_head_dim;
              // Stage k and q of the paired QK head (128 threads == 128 channels).
              sk[thread_x()] = k.read(qk_row + thread_x());
              sq[thread_x()] = q.read(qk_row + thread_x());
              sync_block();
              // Decay the column and compute err = v[v] - sum_k S[k,v] k[k].
              Var<float> dot = def(0.0f);
              for (uint32_t ki = 0u; ki < qwen35_gdn_model_head_dim; ki++) {
                  col[ki] = col[ki] * alpha;
                  dot += col[ki] * sk[ki];
              }
              const auto vv = conv_out.read(v_src_base + v_col);
              const auto err = vv - dot;
              // Rank-1 update S[k,v] += beta * k[k] * err, then o[v] = sum_k S[k,v] q[k].
              Var<float> o = def(0.0f);
              for (uint32_t ki = 0u; ki < qwen35_gdn_model_head_dim; ki++) {
                  col[ki] = col[ki] + bt * sk[ki] * err;
                  o += col[ki] * sq[ki];
              }
              // Write the state column back now: the token's state update is complete and the
              // column is never read again, so its 128 registers die before the barrier +
              // shuffle tree below instead of having to survive them. (Measured cheaper on DX;
              // the per-launch gap that remains there is NOT explained, see the phase-9e report.)
              for (uint32_t ki = 0u; ki < qwen35_gdn_model_head_dim; ki++) {
                  state.write(s_base + ki * qwen35_gdn_model_head_dim + v_col, col[ki]);
              }
              // ---- BLOCK-WIDE gated RMSNorm, in this block -------------------------------
              // Each lane already owns its own output element o, so the reduction is over
              // DISTINCT registers (no shared staging of the 128-channel row at all):
              //   1. warp_active_sum over the 32 lanes of a warp -> the partial of channels
              //      [32w, 32w+32), written to shared by lane 0 of warp w,
              //   2. one sync_block() (the only barrier of the reduction),
              //   3. a fixed-order fold over the four warp partials, identical in every lane,
              // so `inv` is bit-identical across the four warps and no broadcast is needed.
              // This is the block-wide reduction the 4-CTA decode kernel cannot do (its four
              // blocks of a head hold disjoint key slices). Staging the whole row in shared
              // memory instead was measured on all three backends and changed nothing, so the
              // register form (no row staging, no second barrier) is what ships; DX is slow per
              // launch for this kernel either way and the reason was NOT established.
              // 128-float state column and the row out of shared memory: staging the row
              // instead measured +40 % on DX, where the column then spilled.
              Var<float> sumsq = o * o;
              sumsq = warp_active_sum(sumsq);
              $if (lane == 0u) { sred[thread_x() / 32u] = sumsq; };
              sync_block();
              Var<float> total = def(0.0f);
              for (uint32_t w = 0u; w < 4u; w++) { total += sred[w]; }
              const auto inv = rsqrt(total / cast<float>(qwen35_gdn_model_head_dim) + eps);
              // DIRECT norm weight, norm BEFORE the SiLU(z) gate (same expression and
              // operand order as the standalone gated norm kernel).
              out.write(row + v_col,
                  o * inv * norm_w.read(v_col) * qwen_silu(z.read(row + v_col)));
          };
      };
      return kernel;
  }

  [[nodiscard]] Result<QwenLaunch1D> plan_qwen35_gdn_recurrent_norm_dispatch(uint64_t v_heads) {
      if (v_heads == 0u || v_heads > 0xFFFFFFFFull) {
          return Result<QwenLaunch1D>::failure(
              err_invalid("fused GDN recurrent+norm needs 1..uint32 V heads (one block each)"));
      }
      const auto threads = checked_product_u32(v_heads, 128u, "gdn recurrent norm threads");
      if (!threads.ok()) { return Result<QwenLaunch1D>::failure(threads.status); }
      auto plan = plan_qwen_threads(*threads.value, 128u);
      if (!plan.ok()) { return plan; }
      plan.value->work_items = static_cast<uint32_t>(v_heads);
      plan.value->warp_size = 32u;
      return plan;
  }

  [[nodiscard]] Status validate_qwen35_gdn_recurrent_norm_inputs(
      uint64_t tokens, uint64_t v_heads, uint64_t qk_heads, uint64_t head_dim) {
      if (tokens != 1u || head_dim != qwen35_gdn_model_head_dim) {
          return Status::error(StatusCode::unsupported,
                               "fused GDN recurrent+norm is a single-token realization of the "
                               "head_dim=128 model geometry");
      }
      if (v_heads == 0u || qk_heads == 0u || v_heads % qk_heads != 0u) {
          return err_invalid("GDN requires v_heads % qk_heads == 0");
      }
      if (!plan_qwen35_gdn_recurrent_norm_dispatch(v_heads).ok()) {
          return err_overflow("fused GDN recurrent+norm grid exceeds uint32");
      }
      const auto v_n = checked_product_u32(v_heads, head_dim, "GDN recurrent+norm rows");
      if (!v_n.ok() || !checked_product_u32(*v_n.value, head_dim, "GDN recurrent+norm elements").ok()) {
          return err_overflow("GDN recurrent+norm element count exceeds uint32");
      }
      if (!checked_product_u32(2u * qk_heads, head_dim, "GDN recurrent+norm qk").ok()) {
          return err_overflow("GDN recurrent+norm qk element count exceeds uint32");
      }
      return Status::success();
  }

  [[nodiscard]] uint64_t qwen_gdn_recurrent_norm_shared_bytes(uint64_t head_dim) noexcept {
      // sk, sq and so are statically sized at the model head width.
      (void)head_dim;
      return 3u * qwen35_gdn_model_head_dim * sizeof(float);
  }

// ---------------------------------------------------------------------------
// 9. CPU oracles (double accumulation; independent scalar code paths)
// ---------------------------------------------------------------------------
namespace {

[[nodiscard]] double ref_silu(double x) { return x / (1.0 + std::exp(-x)); }
[[nodiscard]] double ref_sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

[[nodiscard]] double ref_decode_weight(QwenWeightFormat format, luisa::span<const uint32_t> words,
                                       luisa::span<const float> scales, uint64_t k, uint32_t group,
                                       uint64_t index) {
    return static_cast<double>(host_decode_packed(format, words, scales, k, group, index));
}

}// namespace

[[nodiscard]] Status ref_qwen_embedding(QwenWeightFormat format, luisa::span<const uint32_t> ids,
                                           luisa::span<const uint32_t> table, luisa::span<const float> scales,
                                           uint64_t hidden, uint64_t vocab, uint32_t group,
                                           luisa::vector<float> &out) {
    const auto v = validate_qwen_embedding_inputs(ids, hidden, vocab);
    if (!v.ok()) { return v; }
    const auto wc = packed_matrix_words(format, hidden, vocab);
    if (!wc.ok()) { return wc.status; }
    if (table.size() != *wc.value) { return err_invalid("embedding oracle: table word count mismatch"); }
    if (format != QwenWeightFormat::bf16) {
        const auto sc = weight_scale_elements(format, hidden, vocab, group);
        if (!sc.ok()) { return sc.status; }
        if (scales.size() != *sc.value) { return err_invalid("embedding oracle: scale count mismatch"); }
    }
    out.resize(ids.size() * static_cast<size_t>(hidden));
    for (size_t t = 0; t < ids.size(); t++) {
        for (uint64_t c = 0; c < hidden; c++) {
            out[t * static_cast<size_t>(hidden) + static_cast<size_t>(c)] =
                host_decode_packed(format, table, scales, hidden, group,
                                   static_cast<uint64_t>(ids[t]) * hidden + c);
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_rmsnorm(luisa::span<const float> x, luisa::span<const float> w,
                                         luisa::span<const float> residual, bool has_residual,
                                         uint32_t rows, uint32_t dim, float eps,
                                         luisa::vector<float> &y, luisa::vector<float> &sum_out) {
    if (dim == 0u) { return err_invalid("rmsnorm dim must be nonzero"); }
    const uint64_t n = static_cast<uint64_t>(rows) * dim;
    if (x.size() != n || w.size() != dim) { return err_invalid("rmsnorm oracle size mismatch"); }
    if (has_residual && residual.size() != n) { return err_invalid("rmsnorm residual size mismatch"); }
    y.resize(static_cast<size_t>(n));
    if (has_residual) { sum_out.resize(static_cast<size_t>(n)); }
    for (uint32_t r = 0; r < rows; r++) {
        const uint64_t base = static_cast<uint64_t>(r) * dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < dim; i++) {
            double v = x[base + i];
            if (has_residual) {
                v += residual[base + i];
                sum_out[base + i] = static_cast<float>(v);
            }
            ss += v * v;
        }
        const double inv = 1.0 / std::sqrt(ss / dim + eps);
        for (uint32_t i = 0; i < dim; i++) {
            const double v = has_residual ? static_cast<double>(sum_out[base + i]) : static_cast<double>(x[base + i]);
            // GGUF weights are pre-added (1+w folded offline); direct multiply.
            y[base + i] = static_cast<float>(v * inv * static_cast<double>(w[i]));
        }
    }
    return Status::success();
}

// Independent oracle for QwenNormEpilogue::snapshot: sum_out is a BIT-EXACT
// copy of the original x, y is the plain direct-pre-added-weight RMSNorm of
// that copy. Deliberately a separate code path from ref_qwen_rmsnorm (plain
// branch) so a snapshot regression cannot hide behind a shared helper bug.
[[nodiscard]] Status ref_qwen_rmsnorm_snapshot(luisa::span<const float> x, luisa::span<const float> w,
                                               uint32_t rows, uint32_t dim, float eps,
                                               luisa::vector<float> &y, luisa::vector<float> &sum_out) {
    if (dim == 0u) { return err_invalid("rmsnorm snapshot dim must be nonzero"); }
    const uint64_t n = static_cast<uint64_t>(rows) * dim;
    if (x.size() != n || w.size() != dim) { return err_invalid("rmsnorm snapshot oracle size mismatch"); }
    y.resize(static_cast<size_t>(n));
    sum_out.assign(x.begin(), x.end());// exact copy, no arithmetic
    for (uint32_t r = 0; r < rows; r++) {
        const uint64_t base = static_cast<uint64_t>(r) * dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < dim; i++) {
            const double v = sum_out[base + i];// normalize FROM the snapshot
            ss += v * v;
        }
        const double inv = 1.0 / std::sqrt(ss / dim + eps);
        for (uint32_t i = 0; i < dim; i++) {
            // GGUF weights are pre-added (1+w folded offline); direct multiply.
            y[base + i] = static_cast<float>(static_cast<double>(sum_out[base + i]) * inv *
                                             static_cast<double>(w[i]));
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_matmul(QwenWeightFormat format, luisa::span<const float> x,
                                        luisa::span<const uint32_t> w, luisa::span<const float> scales,
                                        luisa::span<const float> residual, bool has_residual,
                                        uint32_t m, uint32_t n, uint32_t k, uint32_t group,
                                        luisa::vector<float> &y) {
    if (k == 0u) { return err_invalid("matmul K must be nonzero"); }
    const auto wc = packed_matrix_words(format, k, n);
    if (!wc.ok()) { return wc.status; }
    if (w.size() != *wc.value) { return err_invalid("matmul oracle weight word count mismatch"); }
    if (x.size() != static_cast<uint64_t>(m) * k) { return err_invalid("matmul oracle x size mismatch"); }
    if (has_residual && residual.size() != static_cast<uint64_t>(m) * n) {
        return err_invalid("matmul oracle residual size mismatch");
    }
    if (format != QwenWeightFormat::bf16) {
        const auto sc = weight_scale_elements(format, k, n, group);
        if (!sc.ok()) { return sc.status; }
        if (scales.size() != *sc.value) { return err_invalid("matmul oracle scale count mismatch"); }
    }
    y.resize(static_cast<size_t>(m) * n);
    for (uint32_t mi = 0; mi < m; mi++) {
        for (uint32_t ni = 0; ni < n; ni++) {
            double acc = 0.0;
            for (uint32_t ki = 0; ki < k; ki++) {
                acc += static_cast<double>(x[static_cast<uint64_t>(mi) * k + ki]) *
                       ref_decode_weight(format, w, scales, k, group, static_cast<uint64_t>(ni) * k + ki);
            }
            if (has_residual) { acc += residual[static_cast<uint64_t>(mi) * n + ni]; }
            y[static_cast<size_t>(mi) * n + ni] = static_cast<float>(acc);
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_swiglu(luisa::span<const float> gate, luisa::span<const float> up,
                                        luisa::vector<float> &out) {
    if (gate.size() != up.size()) { return err_invalid("swiglu gate/up size mismatch"); }
    out.resize(gate.size());
    for (size_t i = 0; i < gate.size(); i++) {
        out[i] = static_cast<float>(ref_silu(gate[i]) * static_cast<double>(up[i]));
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_swiglu_fused(luisa::span<const float> proj, uint32_t m, uint32_t f,
                                              luisa::vector<float> &out) {
    if (proj.size() != static_cast<uint64_t>(m) * 2u * f) { return err_invalid("fused swiglu size mismatch"); }
    out.resize(static_cast<size_t>(m) * f);
    for (uint32_t mi = 0; mi < m; mi++) {
        for (uint32_t fi = 0; fi < f; fi++) {
            const uint64_t base = static_cast<uint64_t>(mi) * 2u * f;
            out[static_cast<size_t>(mi) * f + fi] =
                static_cast<float>(ref_silu(proj[base + fi]) * static_cast<double>(proj[base + f + fi]));
        }
    }
    return Status::success();
}

namespace {

// Shared norm+rope row transform for the prepare oracle.
void ref_norm_rope_row(luisa::span<const float> src, luisa::span<const float> nw,
                       float pos, uint32_t head_dim, uint32_t rope_dim, float theta, float eps,
                       luisa::span<float> dst) {
    double ss = 0.0;
    for (uint32_t i = 0; i < head_dim; i++) { ss += static_cast<double>(src[i]) * src[i]; }
    const double inv = 1.0 / std::sqrt(ss / head_dim + eps);
    // Norm first into dst (direct pre-added weight), then NeoX rotate-half
    // on the prefix: pairs (p, p+half), half = rope_dim/2.
    for (uint32_t i = 0; i < head_dim; i++) {
        dst[i] = static_cast<float>(src[i] * inv * static_cast<double>(nw[i]));
    }
    const uint32_t half = rope_dim / 2u;
    for (uint32_t p = 0; p < half; p++) {
        const double freq = std::pow(static_cast<double>(theta), -static_cast<double>(2u * p) / rope_dim);
        const double angle = static_cast<double>(pos) * freq;
        const double c = std::cos(angle), s = std::sin(angle);
        const double x0 = dst[p], x1 = dst[p + half];
        dst[p] = static_cast<float>(x0 * c - x1 * s);
        dst[p + half] = static_cast<float>(x0 * s + x1 * c);
    }
}

}// namespace

[[nodiscard]] Status ref_qwen_attn_prepare(luisa::span<const float> q_proj, luisa::span<const float> k_proj,
                                              luisa::span<const float> v_proj, luisa::span<const float> q_norm_w,
                                              luisa::span<const float> k_norm_w, luisa::span<const int32_t> positions,
                                              uint32_t tokens, const QwenDslConfig &cfg, uint32_t pos0,
                                              bool append_kv, luisa::span<float> k_cache, luisa::span<float> v_cache,
                                              luisa::vector<float> &q, luisa::vector<float> &gate,
                                              luisa::vector<float> &k, luisa::vector<float> &v) {
    const auto val = validate_qwen_attn_prepare_inputs(tokens, cfg, pos0,
                                                       append_kv ? k_cache.size() / cfg.attn_kv_n() : 0u, append_kv);
    if (!val.ok()) { return val; }
    const uint32_t hd = cfg.attn_head_dim;
    const uint64_t qn = static_cast<uint64_t>(tokens) * cfg.attn_q_proj_n();
    const uint64_t kvn = static_cast<uint64_t>(tokens) * cfg.attn_kv_n();
    if (q_proj.size() != qn || k_proj.size() != kvn || v_proj.size() != kvn ||
        q_norm_w.size() != hd || k_norm_w.size() != hd || positions.size() != tokens) {
        return err_invalid("attn prepare oracle size mismatch");
    }
    if (append_kv) {
        const uint64_t cap = k_cache.size() / cfg.attn_kv_n();
        if (k_cache.size() != cap * cfg.attn_kv_n() || v_cache.size() != k_cache.size()) {
            return err_invalid("attn prepare oracle cache size mismatch");
        }
    }
    q.resize(static_cast<size_t>(tokens) * cfg.q_heads * hd);
    gate.resize(q.size());
    k.resize(static_cast<size_t>(kvn));
    v.resize(static_cast<size_t>(kvn));
    for (uint32_t t = 0; t < tokens; t++) {
        const float pos = static_cast<float>(positions[t]);
        for (uint32_t h = 0; h < cfg.q_heads; h++) {
            const uint64_t src = (static_cast<uint64_t>(t) * cfg.q_heads + h) * 2u * hd;
            const uint64_t dst = (static_cast<uint64_t>(t) * cfg.q_heads + h) * hd;
            ref_norm_rope_row(q_proj.subspan(src, hd), q_norm_w, pos, hd, cfg.rope_dim,
                              cfg.rope_theta, cfg.norm_eps, luisa::span<float>{q}.subspan(dst, hd));
            for (uint32_t i = 0; i < hd; i++) { gate[dst + i] = q_proj[src + hd + i]; }
        }
        for (uint32_t h = 0; h < cfg.kv_heads; h++) {
            const uint64_t off = (static_cast<uint64_t>(t) * cfg.kv_heads + h) * hd;
            ref_norm_rope_row(k_proj.subspan(off, hd), k_norm_w, pos, hd, cfg.rope_dim,
                              cfg.rope_theta, cfg.norm_eps, luisa::span<float>{k}.subspan(off, hd));
            for (uint32_t i = 0; i < hd; i++) { v[off + i] = v_proj[off + i]; }
            if (append_kv) {
                const uint64_t slot = (static_cast<uint64_t>(pos0) + t) * cfg.attn_kv_n() + h * hd;
                for (uint32_t i = 0; i < hd; i++) {
                    k_cache[slot + i] = k[off + i];
                    v_cache[slot + i] = v[off + i];
                }
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_rope_partial(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                              uint32_t tokens, uint32_t heads, uint32_t head_dim,
                                              uint32_t rope_dim, float theta, luisa::vector<float> &out) {
    if (rope_dim % 2u != 0u || rope_dim > head_dim) { return err_invalid("rope_dim must be even and <= head_dim"); }
    const uint64_t n = static_cast<uint64_t>(tokens) * heads * head_dim;
    if (x.size() != n || positions.size() != tokens) { return err_invalid("rope oracle size mismatch"); }
    out.resize(static_cast<size_t>(n));
    for (uint32_t t = 0; t < tokens; t++) {
        const double pos = positions[t];
        for (uint32_t h = 0; h < heads; h++) {
            const uint64_t base = (static_cast<uint64_t>(t) * heads + h) * head_dim;
            for (uint32_t i = 0; i < head_dim; i++) { out[base + i] = x[base + i]; }
            // NeoX rotate-half: pairs (p, p+half), duplicated cos/sin.
            const uint32_t half = rope_dim / 2u;
            for (uint32_t p = 0; p < half; p++) {
                const double freq = std::pow(static_cast<double>(theta), -static_cast<double>(2u * p) / rope_dim);
                const double angle = pos * freq;
                const double c = std::cos(angle), s = std::sin(angle);
                const double x0 = x[base + p], x1 = x[base + p + half];
                out[base + p] = static_cast<float>(x0 * c - x1 * s);
                out[base + p + half] = static_cast<float>(x0 * s + x1 * c);
            }
        }
    }
    return Status::success();
}

namespace {

// Naive two-pass causal GQA attention over the cache (oracle core).
[[nodiscard]] Status ref_attn_core(luisa::span<const float> q, luisa::span<const float> gate,
                                      luisa::span<const float> k_cache, luisa::span<const float> v_cache,
                                      uint32_t tokens, uint32_t q_heads, uint32_t kv_heads,
                                      uint32_t head_dim, float scale, uint32_t pos0, uint32_t kv_len,
                                      luisa::vector<float> &out) {
    if (q_heads == 0u || kv_heads == 0u || q_heads % kv_heads != 0u) {
        return err_invalid("attention oracle GQA ratio invalid");
    }
    const uint64_t qn = static_cast<uint64_t>(tokens) * q_heads * head_dim;
    if (q.size() != qn || gate.size() != qn) { return err_invalid("attention oracle q/gate size mismatch"); }
    const uint64_t kv_stride = static_cast<uint64_t>(kv_heads) * head_dim;
    if (k_cache.size() < static_cast<uint64_t>(kv_len) * kv_stride ||
        v_cache.size() < static_cast<uint64_t>(kv_len) * kv_stride) {
        return err_invalid("attention oracle cache too small for kv_len");
    }
    out.resize(static_cast<size_t>(qn));
    luisa::vector<double> scores(kv_len);
    for (uint32_t t = 0; t < tokens; t++) {
        const uint32_t k_end = pos0 + t + 1u;
        if (k_end > kv_len) { return err_invalid("attention oracle causal window exceeds kv_len"); }
        for (uint32_t h = 0; h < q_heads; h++) {
            const uint32_t kvh = h / (q_heads / kv_heads);
            const uint64_t q_base = (static_cast<uint64_t>(t) * q_heads + h) * head_dim;
            double max_s = -1e300;
            for (uint32_t key = 0; key < k_end; key++) {
                const uint64_t k_base = static_cast<uint64_t>(key) * kv_stride + kvh * head_dim;
                double s = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    s += static_cast<double>(q[q_base + d]) * k_cache[k_base + d];
                }
                s *= scale;
                scores[key] = s;
                if (s > max_s) { max_s = s; }
            }
            double sum = 0.0;
            for (uint32_t key = 0; key < k_end; key++) {
                scores[key] = std::exp(scores[key] - max_s);
                sum += scores[key];
            }
            for (uint32_t d = 0; d < head_dim; d++) {
                double o = 0.0;
                for (uint32_t key = 0; key < k_end; key++) {
                    const uint64_t v_base = static_cast<uint64_t>(key) * kv_stride + kvh * head_dim;
                    o += scores[key] * v_cache[v_base + d];
                }
                out[q_base + d] = static_cast<float>(o / sum * ref_sigmoid(gate[q_base + d]));
            }
        }
    }
    return Status::success();
}

}// namespace

[[nodiscard]] Status ref_qwen_attn_prefill(luisa::span<const float> q, luisa::span<const float> gate,
                                              luisa::span<const float> k_cache, luisa::span<const float> v_cache,
                                              uint32_t tokens, uint32_t q_heads, uint32_t kv_heads,
                                              uint32_t head_dim, float scale, uint32_t pos0, uint32_t kv_len,
                                              luisa::vector<float> &out) {
    return ref_attn_core(q, gate, k_cache, v_cache, tokens, q_heads, kv_heads, head_dim, scale, pos0, kv_len, out);
}

[[nodiscard]] Status ref_qwen_attn_decode(luisa::span<const float> q, luisa::span<const float> gate,
                                             luisa::span<const float> k_cache, luisa::span<const float> v_cache,
                                             uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                                             float scale, uint32_t kv_len, luisa::vector<float> &out) {
    if (kv_len == 0u) { return err_invalid("decode oracle requires kv_len >= 1"); }
    // The decode query sits at absolute position kv_len-1.
    return ref_attn_core(q, gate, k_cache, v_cache, 1u, q_heads, kv_heads, head_dim, scale,
                         kv_len - 1u, kv_len, out);
}

[[nodiscard]] Status ref_qwen_gdn_conv(luisa::span<const float> src, luisa::span<const float> weight,
                                          luisa::span<float> ring, uint32_t tokens, uint32_t channels,
                                          uint32_t kernel_size, uint32_t pos0, luisa::vector<float> &out) {
    const auto val = validate_qwen_gdn_conv_inputs(tokens, channels, kernel_size, pos0);
    if (!val.ok()) { return val; }
    const uint64_t rw = kernel_size - 1u;
    if (src.size() != static_cast<uint64_t>(tokens) * channels ||
        weight.size() != static_cast<uint64_t>(channels) * kernel_size ||
        ring.size() != static_cast<uint64_t>(channels) * rw) {
        return err_invalid("gdn conv oracle size mismatch");
    }
    out.resize(static_cast<size_t>(tokens) * channels);
    for (uint32_t t = 0; t < tokens; t++) {
        const uint64_t p = static_cast<uint64_t>(pos0) + t;
        for (uint32_t c = 0; c < channels; c++) {
            double acc = 0.0;
            for (uint32_t j = 0; j < kernel_size; j++) {
                const int64_t abs_pos = static_cast<int64_t>(p) - static_cast<int64_t>(rw) + j;
                double xv = 0.0;
                if (abs_pos >= 0) {
                    if (abs_pos < pos0) {
                        xv = ring[static_cast<size_t>(c) * rw + static_cast<size_t>(abs_pos % static_cast<int64_t>(rw))];
                    } else {
                        xv = src[static_cast<size_t>(abs_pos - pos0) * channels + c];
                    }
                }
                // GGUF ssm_conv1d payload is channel-major: weight[c*K + j].
                acc += static_cast<double>(weight[static_cast<size_t>(c) * kernel_size + j]) * xv;
            }
            out[static_cast<size_t>(t) * channels + c] = static_cast<float>(ref_silu(acc));
        }
        for (uint32_t c = 0; c < channels; c++) {
            ring[static_cast<size_t>(c) * rw + static_cast<size_t>(p % rw)] = src[static_cast<size_t>(t) * channels + c];
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_gdn_gates(luisa::span<const float> a, luisa::span<const float> b,
                                           luisa::span<const float> ssm_a, luisa::span<const float> dt_bias,
                                           uint32_t tokens, uint32_t heads, float softplus_threshold,
                                           luisa::vector<float> &beta, luisa::vector<float> &g) {
    if (heads == 0u) { return err_invalid("gdn gates heads must be nonzero"); }
    const uint64_t n = static_cast<uint64_t>(tokens) * heads;
    if (a.size() != n || b.size() != n || ssm_a.size() != heads || dt_bias.size() != heads) {
        return err_invalid("gdn gates oracle size mismatch");
    }
    beta.resize(static_cast<size_t>(n));
    g.resize(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; i++) {
        const uint32_t h = static_cast<uint32_t>(i % heads);
        const double av = static_cast<double>(a[i]) + dt_bias[h];
        const double sp = av > softplus_threshold ? av : std::log1p(std::exp(av));
        beta[static_cast<size_t>(i)] = static_cast<float>(ref_sigmoid(b[i]));
        // GGUF ssm_a is already -exp(A_log) (negative); direct multiply.
        g[static_cast<size_t>(i)] = static_cast<float>(static_cast<double>(ssm_a[h]) * sp);
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_gdn_qk_norm(luisa::span<const float> conv_out, uint32_t tokens,
                                             uint32_t qk_heads, uint32_t head_dim, uint32_t v_channels,
                                             float eps, float q_scale,
                                             luisa::vector<float> &q, luisa::vector<float> &k) {
    if (qk_heads == 0u || head_dim == 0u) { return err_invalid("gdn qk norm geometry must be nonzero"); }
    const uint64_t stride = 2u * static_cast<uint64_t>(qk_heads) * head_dim + v_channels;
    if (conv_out.size() != static_cast<uint64_t>(tokens) * stride) {
        return err_invalid("gdn qk norm oracle conv_out size mismatch");
    }
    const uint64_t n = static_cast<uint64_t>(tokens) * qk_heads * head_dim;
    q.resize(static_cast<size_t>(n));
    k.resize(static_cast<size_t>(n));
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < qk_heads; h++) {
            const uint64_t q_src = static_cast<uint64_t>(t) * stride + h * head_dim;
            const uint64_t k_src = q_src + static_cast<uint64_t>(qk_heads) * head_dim;
            const uint64_t dst = (static_cast<uint64_t>(t) * qk_heads + h) * head_dim;
            double qss = 0.0, kss = 0.0;
            for (uint32_t i = 0; i < head_dim; i++) {
                qss += static_cast<double>(conv_out[q_src + i]) * conv_out[q_src + i];
                kss += static_cast<double>(conv_out[k_src + i]) * conv_out[k_src + i];
            }
            // FLA/HF l2norm: eps inside the rsqrt.
            const double q_inv = 1.0 / std::sqrt(qss + static_cast<double>(eps));
            const double k_inv = 1.0 / std::sqrt(kss + static_cast<double>(eps));
            for (uint32_t i = 0; i < head_dim; i++) {
                q[dst + i] = static_cast<float>(conv_out[q_src + i] * q_inv * q_scale);
                k[dst + i] = static_cast<float>(conv_out[k_src + i] * k_inv);
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_gdn_recurrent(luisa::span<const float> q, luisa::span<const float> k,
                                               luisa::span<const float> conv_out, luisa::span<const float> beta,
                                               luisa::span<const float> g, luisa::span<float> state,
                                               uint32_t tokens, uint32_t v_heads, uint32_t head_dim,
                                               uint32_t qk_heads, luisa::vector<float> &out) {
    const auto val = validate_qwen_gdn_recurrent_inputs(tokens, v_heads, qk_heads, head_dim);
    if (!val.ok()) { return val; }
    const uint64_t qn = static_cast<uint64_t>(tokens) * qk_heads * head_dim;
    const uint64_t vn = static_cast<uint64_t>(tokens) * v_heads * head_dim;
    const uint64_t sn = static_cast<uint64_t>(v_heads) * head_dim * head_dim;
    const uint64_t v_offset = 2u * static_cast<uint64_t>(qk_heads) * head_dim;
    const uint64_t c_stride = v_offset + static_cast<uint64_t>(v_heads) * head_dim;
    if (q.size() != qn || k.size() != qn || conv_out.size() != static_cast<uint64_t>(tokens) * c_stride ||
        beta.size() != vn / head_dim || g.size() != vn / head_dim || state.size() != sn) {
        return err_invalid("gdn recurrent oracle size mismatch");
    }
    out.resize(static_cast<size_t>(vn));
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < v_heads; h++) {
            // Tiled GGUF V order: V head h pairs with QK head h % qk_heads.
            const uint32_t qkh = h % qk_heads;
            const uint64_t th = static_cast<uint64_t>(t) * v_heads + h;
            const uint64_t qk_row = (static_cast<uint64_t>(t) * qk_heads + qkh) * head_dim;
            const uint64_t v_src = static_cast<uint64_t>(t) * c_stride + v_offset + h * head_dim;
            const uint64_t sb = static_cast<uint64_t>(h) * head_dim * head_dim;
            const double alpha = std::exp(static_cast<double>(g[th]));
            const double bt = beta[th];
            for (uint64_t i = 0; i < static_cast<uint64_t>(head_dim) * head_dim; i++) {
                state[sb + i] = static_cast<float>(static_cast<double>(state[sb + i]) * alpha);
            }
            for (uint32_t vi = 0; vi < head_dim; vi++) {
                double sk = 0.0;
                for (uint32_t ki = 0; ki < head_dim; ki++) {
                    sk += static_cast<double>(state[sb + static_cast<uint64_t>(ki) * head_dim + vi]) * k[qk_row + ki];
                }
                const double err = static_cast<double>(conv_out[v_src + vi]) - sk;
                for (uint32_t ki = 0; ki < head_dim; ki++) {
                    const uint64_t idx = sb + static_cast<uint64_t>(ki) * head_dim + vi;
                    state[idx] = static_cast<float>(static_cast<double>(state[idx]) + bt * k[qk_row + ki] * err);
                }
            }
            for (uint32_t vi = 0; vi < head_dim; vi++) {
                double o = 0.0;
                for (uint32_t ki = 0; ki < head_dim; ki++) {
                    o += static_cast<double>(state[sb + static_cast<uint64_t>(ki) * head_dim + vi]) * q[qk_row + ki];
                }
                out[(static_cast<uint64_t>(t) * v_heads + h) * head_dim + vi] = static_cast<float>(o);
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_qwen_gdn_gated_norm(luisa::span<const float> o, luisa::span<const float> z,
                                                luisa::span<const float> norm_w, uint32_t tokens,
                                                uint32_t heads, uint32_t head_dim, float eps,
                                                luisa::vector<float> &out) {
    if (heads == 0u || head_dim == 0u) { return err_invalid("gdn gated norm geometry must be nonzero"); }
    const uint64_t n = static_cast<uint64_t>(tokens) * heads * head_dim;
    if (o.size() != n || z.size() != n || norm_w.size() != head_dim) {
        return err_invalid("gdn gated norm oracle size mismatch");
    }
    out.resize(static_cast<size_t>(n));
    for (uint64_t r = 0; r < static_cast<uint64_t>(tokens) * heads; r++) {
        const uint64_t base = r * head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; i++) { ss += static_cast<double>(o[base + i]) * o[base + i]; }
        const double inv = 1.0 / std::sqrt(ss / head_dim + eps);
        for (uint32_t i = 0; i < head_dim; i++) {
            out[base + i] = static_cast<float>(o[base + i] * inv * static_cast<double>(norm_w[i]) *
                                               ref_silu(z[base + i]));
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// 10. Device-free self-test
// ---------------------------------------------------------------------------
namespace {

struct SelfTestRng {
    uint64_t state{0x9E3779B97F4A7C15ull};
    [[nodiscard]] uint32_t next() noexcept {
        state ^= state >> 12u;
        state ^= state << 25u;
        state ^= state >> 27u;
        return static_cast<uint32_t>((state * 0x2545F4914F6CDD1Dull) >> 32u);
    }
    [[nodiscard]] float uniform(float lo, float hi) noexcept {
        return lo + (hi - lo) * (static_cast<float>(next() >> 8u) / 16777216.0f);
    }
};

[[nodiscard]] bool close_enough(float a, float b, float rel, float abs) noexcept {
    const double d = std::fabs(static_cast<double>(a) - b);
    return d <= abs + rel * std::fabs(static_cast<double>(b));
}

struct TestCtx {
    luisa::vector<luisa::string> *report;
    uint32_t failures{0};
    void check(bool ok, const char *name) {
        if (report != nullptr) {
            report->push_back(luisa::string{ok ? "PASS " : "FAIL "} + name);
        }
        if (!ok) { failures++; }
    }
};

}// namespace

[[nodiscard]] Status dsl_ops_self_test(luisa::vector<luisa::string> *report) {
    TestCtx ctx{report, 0u};
    SelfTestRng rng;

    // --- host bit helpers: bf16 round-trip of exactly representable values.
    {
        bool ok = true;
        for (int i = 0; i < 1000; i++) {
            const float v = rng.uniform(-100.0f, 100.0f);
            const uint16_t b = host_f32_to_bf16_bits(v);
            const float back = host_bf16_bits_to_f32(b);
            // bf16 has 8 mantissa bits: relative error <= 2^-8.
            ok = ok && close_enough(back, v, 1.0f / 128.0f, 1e-30f);
        }
        // Exact values round-trip bit-exactly.
        ok = ok && host_bf16_bits_to_f32(host_f32_to_bf16_bits(1.5f)) == 1.5f;
        ok = ok && host_bf16_bits_to_f32(host_f32_to_bf16_bits(-0.25f)) == -0.25f;
        ctx.check(ok, "bf16 host round-trip");
    }
    // --- fp8 E4M3: known codes, subnormals, saturation, NaN.
    {
        bool ok = true;
        ok = ok && host_fp8_e4m3_bits_to_f32(0x00u) == 0.0f;
        ok = ok && host_fp8_e4m3_bits_to_f32(0x80u) == 0.0f;// -0
        ok = ok && host_fp8_e4m3_bits_to_f32(0x38u) == 1.0f;// e=7,m=0
        ok = ok && host_fp8_e4m3_bits_to_f32(0x7Eu) == 448.0f;// max finite
        ok = ok && host_fp8_e4m3_bits_to_f32(0x01u) == std::ldexp(1.0f, -9);// min subnormal
        ok = ok && std::isnan(host_fp8_e4m3_bits_to_f32(0x7Fu));
        ok = ok && host_f32_to_fp8_e4m3_bits(1.0f) == 0x38u;
        ok = ok && host_f32_to_fp8_e4m3_bits(448.0f) == 0x7Eu;
        ok = ok && host_f32_to_fp8_e4m3_bits(1000.0f) == 0x7Eu;// saturate
        ok = ok && host_f32_to_fp8_e4m3_bits(-1000.0f) == 0xFEu;
        ok = ok && host_f32_to_fp8_e4m3_bits(0.0f) == 0x00u;
        // RNE tie: 1.0625 is exactly between 1.0 (m=0) and 1.125 (m=1) -> even m=0.
        ok = ok && host_f32_to_fp8_e4m3_bits(1.0625f) == 0x38u;
        // Round-trip of all finite codes.
        for (uint32_t c = 0; c < 256u; c++) {
            if ((c & 0x7Fu) == 0x7Fu) { continue; }// NaN
            const float v = host_fp8_e4m3_bits_to_f32(static_cast<uint8_t>(c));
            ok = ok && host_f32_to_fp8_e4m3_bits(v) == static_cast<uint8_t>(c);
        }
        ctx.check(ok, "fp8 e4m3 host codec");
    }
    // --- packing + matmul oracle vs an independent dequantized reference,
    //     ragged K (odd for fp8 partial groups/words), small M/N.
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        const uint32_t K = 70u, N = 5u, M = 3u;// K=70: partial fp8 group (64+6), odd word tail
        luisa::vector<float> w(static_cast<size_t>(K) * N), x(static_cast<size_t>(M) * K);
        for (auto &v : w) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : x) { v = rng.uniform(-1.0f, 1.0f); }
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        const auto pack = host_pack_matrix(format, w, K, N, 64u, words, scales);
        ctx.check(pack.ok(), format == QwenWeightFormat::bf16 ? "pack bf16 ragged" : "pack fp8 ragged");
        // Independent reference: dequantize via host_decode_packed, double dot.
        luisa::vector<float> expect(static_cast<size_t>(M) * N);
        for (uint32_t mi = 0; mi < M; mi++) {
            for (uint32_t ni = 0; ni < N; ni++) {
                double acc = 0.0;
                for (uint32_t ki = 0; ki < K; ki++) {
                    acc += static_cast<double>(x[mi * K + ki]) *
                           host_decode_packed(format, words, scales, K, 64u, ni * K + ki);
                }
                expect[mi * N + ni] = static_cast<float>(acc);
            }
        }
        luisa::vector<float> y;
        const auto mm = ref_qwen_matmul(format, x, words, scales, {}, false, M, N, K, 64u, y);
        bool ok = mm.ok() && y.size() == expect.size();
        const float tol = format == QwenWeightFormat::bf16 ? 2e-2f : 0.35f;
        for (size_t i = 0; ok && i < y.size(); i++) { ok = close_enough(y[i], expect[i], tol, 1e-4f); }
        ctx.check(ok, format == QwenWeightFormat::bf16 ? "matmul bf16 oracle ragged" : "matmul fp8 oracle ragged");
        // Zero group: all-zero weights must give scale 0 and exact zero output.
        luisa::vector<float> zw(static_cast<size_t>(K) * N, 0.0f);
        luisa::vector<uint32_t> zw_words;
        luisa::vector<float> zw_scales;
        const auto zpack = host_pack_matrix(format, zw, K, N, 64u, zw_words, zw_scales);
        luisa::vector<float> zy;
        const auto zmm = ref_qwen_matmul(format, x, zw_words, zw_scales, {}, false, M, N, K, 64u, zy);
        bool zok = zpack.ok() && zmm.ok();
        for (auto v : zy) { zok = zok && v == 0.0f; }
        if (format == QwenWeightFormat::fp8_e4m3) {
            for (auto s : zw_scales) { zok = zok && s == 0.0f; }
        }
        ctx.check(zok, format == QwenWeightFormat::bf16 ? "matmul bf16 zero group" : "matmul fp8 zero group");
    }
    // --- embedding gather oracle (both formats, ragged hidden).
    for (auto format : {QwenWeightFormat::bf16, QwenWeightFormat::fp8_e4m3}) {
        const uint64_t hidden = 34u, vocab = 7u;// hidden=34: even for bf16, partial fp8 group
        luisa::vector<float> table(static_cast<size_t>(hidden * vocab));
        for (auto &v : table) { v = rng.uniform(-1.0f, 1.0f); }
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        const auto pack = host_pack_matrix(format, table, hidden, vocab, 64u, words, scales);
        const luisa::vector<uint32_t> ids{6u, 0u, 3u, 6u};
        luisa::vector<float> out;
        const auto emb = ref_qwen_embedding(format, ids, words, scales, hidden, vocab, 64u, out);
        bool ok = pack.ok() && emb.ok() && out.size() == ids.size() * hidden;
        for (size_t t = 0; ok && t < ids.size(); t++) {
            for (uint64_t c = 0; c < hidden; c++) {
                const float expect = host_decode_packed(format, words, scales, hidden, 64u, ids[t] * hidden + c);
                ok = close_enough(out[t * hidden + c], expect, 0.0f, 0.0f);
            }
        }
        ctx.check(ok, format == QwenWeightFormat::bf16 ? "embedding bf16 oracle" : "embedding fp8 oracle");
        // Out-of-range id must be rejected.
        const luisa::vector<uint32_t> bad{7u};
        luisa::vector<float> dummy;
        ctx.check(!ref_qwen_embedding(format, bad, words, scales, hidden, vocab, 64u, dummy).ok(),
                  "embedding rejects out-of-range id");
    }
    // --- rmsnorm (plain + residual), ragged dim.
    {
        const uint32_t rows = 3u, dim = 37u;
        luisa::vector<float> x(rows * dim), w(dim), res(rows * dim);
        for (auto &v : x) { v = rng.uniform(-2.0f, 2.0f); }
        // GGUF norm weights are PRE-ADDED (converter folds 1+w offline):
        // generate them around 1, exactly like the real checkpoint.
        for (auto &v : w) { v = rng.uniform(0.5f, 1.5f); }
        for (auto &v : res) { v = rng.uniform(-1.0f, 1.0f); }
        luisa::vector<float> y, sum;
        const auto r1 = ref_qwen_rmsnorm(x, w, {}, false, rows, dim, 1e-6f, y, sum);
        // Independent scalar check of row 0 (direct weight multiply).
        double ss = 0.0;
        for (uint32_t i = 0; i < dim; i++) { ss += static_cast<double>(x[i]) * x[i]; }
        const double inv = 1.0 / std::sqrt(ss / dim + 1e-6);
        bool ok = r1.ok();
        for (uint32_t i = 0; ok && i < dim; i++) {
            ok = close_enough(y[i], static_cast<float>(x[i] * inv * w[i]), 1e-5f, 1e-6f);
        }
        ctx.check(ok, "rmsnorm pre-added direct-weight oracle");
        const auto r2 = ref_qwen_rmsnorm(x, w, res, true, rows, dim, 1e-6f, y, sum);
        ok = r2.ok();
        double ss2 = 0.0;
        for (uint32_t i = 0; i < dim; i++) {
            const double v = static_cast<double>(x[i]) + res[i];
            ss2 += v * v;
        }
        const double inv2 = 1.0 / std::sqrt(ss2 / dim + 1e-6);
        for (uint32_t i = 0; ok && i < dim; i++) {
            ok = close_enough(y[i], static_cast<float>((x[i] + res[i]) * inv2 * w[i]), 1e-5f, 1e-6f);
        }
        ctx.check(ok, "rmsnorm residual oracle");
    }
    // --- native Q8_0 host contract (phase 8a): block split, exact decode,
    // geometry gates and the oracle. K = 2560 keeps the model geometry; the
    // codes and FP16 block scales are synthetic (the loader only ever copies
    // them).
    {
        const uint32_t K = 2560u, N = 3u;
        const uint64_t blocks = static_cast<uint64_t>(K) * N / kQwenQ8BlockSize;
        luisa::vector<uint8_t> codes(static_cast<size_t>(K) * N);
        luisa::vector<uint16_t> bits(static_cast<size_t>(blocks));
        for (auto &c : codes) { c = static_cast<uint8_t>(rng.next() & 0xFFu); }
        for (auto &b : bits) { b = f32_to_fp16(rng.uniform(0.001f, 0.02f)); }
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        const auto pack = host_pack_q8_0_codes(codes, bits, K, N, words, scales);
        bool ok = pack.ok() && words.size() == static_cast<size_t>(K) * N / 4u &&
                  scales.size() == static_cast<size_t>(blocks);
        // The codes are a byte copy: element i sits in byte (i % 4) of word i/4.
        for (uint64_t i = 0; ok && i < static_cast<uint64_t>(K) * N; i++) {
            ok = ((words[static_cast<size_t>(i / 4u)] >> (8u * (i % 4u))) & 0xFFu) ==
                 static_cast<uint32_t>(codes[static_cast<size_t>(i)]);
        }
        // Scales are the exact FP16 widening, one per 32-element block.
        for (auto b = 0u; ok && b < bits.size(); b++) {
            ok = scales[b] == fp16_to_f32(bits[b]);
        }
        // Device-parity decode: int8(code) * scale, computed independently here.
        for (uint64_t i = 0; ok && i < static_cast<uint64_t>(K) * N; i++) {
            const float want = int8_to_f32(codes[static_cast<size_t>(i)]) *
                               fp16_to_f32(bits[static_cast<size_t>(i / kQwenQ8BlockSize)]);
            ok = host_decode_packed(QwenWeightFormat::q8_0, words, scales, K, kQwenQ8BlockSize, i) == want;
        }
        ctx.check(ok, "q8_0 host pack/decode parity");
        // Oracle equality against an independent double-accumulation reference
        // built from the same codes (no quantization involved).
        const uint32_t M = 2u;
        luisa::vector<float> x(static_cast<size_t>(M) * K);
        for (auto &v : x) { v = rng.uniform(-1.0f, 1.0f); }
        luisa::vector<float> y;
        const auto mm = ref_qwen_matmul(QwenWeightFormat::q8_0, x, words, scales, {}, false,
                                        M, N, K, kQwenQ8BlockSize, y);
        bool mm_ok = mm.ok() && y.size() == static_cast<size_t>(M) * N;
        for (uint32_t mi = 0; mm_ok && mi < M; mi++) {
            for (uint32_t ni = 0; mm_ok && ni < N; ni++) {
                double acc = 0.0;
                for (uint32_t ki = 0; ki < K; ki++) {
                    const uint64_t idx = static_cast<uint64_t>(ni) * K + ki;
                    acc += static_cast<double>(x[static_cast<uint64_t>(mi) * K + ki]) *
                           static_cast<double>(int8_to_f32(codes[static_cast<size_t>(idx)])) *
                           static_cast<double>(fp16_to_f32(bits[static_cast<size_t>(idx / kQwenQ8BlockSize)]));
                }
                mm_ok = close_enough(y[static_cast<size_t>(mi) * N + ni],
                                     static_cast<float>(acc), 1e-4f, 1e-4f);
            }
        }
        ctx.check(mm_ok, "matmul q8_0 oracle");
        // Embedding oracle over the same matrix (row gather + decode).
        const luisa::vector<uint32_t> ids{2u, 0u, 2u};
        luisa::vector<float> emb;
        bool emb_ok = ref_qwen_embedding(QwenWeightFormat::q8_0, ids, words, scales, K, N,
                                         kQwenQ8BlockSize, emb).ok() &&
                      emb.size() == ids.size() * K;
        for (size_t t = 0; emb_ok && t < ids.size(); t++) {
            for (uint32_t c = 0; emb_ok && c < K; c++) {
                emb_ok = emb[t * K + c] ==
                         host_decode_packed(QwenWeightFormat::q8_0, words, scales, K,
                                            kQwenQ8BlockSize,
                                            static_cast<uint64_t>(ids[t]) * K + c);
            }
        }
        ctx.check(emb_ok, "embedding q8_0 oracle");
        // Geometry / planner fail-closed paths (never a silent fallback).
        ctx.check(!packed_matrix_words(QwenWeightFormat::q8_0, 2540u, 4u).ok(),
                  "q8_0 packed words reject K % 32 != 0");
        ctx.check(packed_matrix_words(QwenWeightFormat::q8_0, 2560u, 0u).ok() &&
                      *packed_matrix_words(QwenWeightFormat::q8_0, 2560u, 0u).value == 0u,
                  "q8_0 packed words: an empty matrix has no words");
        ctx.check(!packed_matrix_words(QwenWeightFormat::q8_0, 2560u, 1ull << 32u).ok(),
                  "q8_0 packed words reject a uint32 overflow");
        ctx.check(q8_0_scale_elements(2560u, 3u).ok() &&
                      *q8_0_scale_elements(2560u, 3u).value == 3u * 80u,
                  "q8_0 scale elements accept a valid shape");
        ctx.check(!weight_scale_elements(QwenWeightFormat::q8_0, 2540u, 4u, kQwenQ8BlockSize).ok(),
                  "q8_0 weight scale elements reject K % 32 != 0");
        ctx.check(!validate_qwen_q8_0_geometry(2560u, 64u).ok(),
                  "q8_0 geometry rejects the FP8 group");
        ctx.check(!validate_qwen_q8_0_geometry(0u, kQwenQ8BlockSize).ok(),
                  "q8_0 geometry rejects K = 0");
        luisa::vector<uint32_t> hw;
        luisa::vector<float> hs;
        luisa::vector<float> ones(static_cast<size_t>(K) * N, 1.0f);
        ctx.check(!host_pack_matrix(QwenWeightFormat::q8_0, ones, K, N, kQwenQ8BlockSize,
                                    hw, hs).ok(),
                  "host_pack_matrix refuses q8_0 re-quantization");
        // A scale buffer that does not match the block count fails closed.
        ctx.check(!ref_qwen_matmul(QwenWeightFormat::q8_0, x, words,
                                   luisa::span<const float>{scales.data(), scales.size() - 1u},
                                   {}, false, M, N, K, kQwenQ8BlockSize, y).ok(),
                  "q8_0 oracle rejects a truncated scale array");
    }
    // --- rmsnorm snapshot: sum_out is x bit-exactly; y == plain norm math.
    {
        const uint32_t rows = 3u, dim = 37u;// ragged: not a multiple of 32
        luisa::vector<float> x(rows * dim), w(dim);
        for (auto &v : x) { v = rng.uniform(-2.0f, 2.0f); }
        for (auto &v : w) { v = rng.uniform(0.5f, 1.5f); }// pre-added GGUF weights
        luisa::vector<float> y, sum;
        const auto r = ref_qwen_rmsnorm_snapshot(x, w, rows, dim, 1e-6f, y, sum);
        bool ok = r.ok() && sum.size() == x.size() && y.size() == x.size();
        for (size_t i = 0; ok && i < x.size(); i++) { ok = sum[i] == x[i]; }// bit-exact
        // Independent scalar check of every row (direct weight multiply).
        for (uint32_t row = 0; ok && row < rows; row++) {
            double ss = 0.0;
            for (uint32_t i = 0; i < dim; i++) {
                const double v = x[row * dim + i];
                ss += v * v;
            }
            const double inv = 1.0 / std::sqrt(ss / dim + 1e-6);
            for (uint32_t i = 0; ok && i < dim; i++) {
                ok = close_enough(y[row * dim + i],
                                  static_cast<float>(x[row * dim + i] * inv * w[i]), 1e-5f, 1e-6f);
            }
        }
        // y must equal the PLAIN oracle exactly (same inputs, same math).
        luisa::vector<float> py, psum;
        const auto rp = ref_qwen_rmsnorm(x, w, {}, false, rows, dim, 1e-6f, py, psum);
        ok = ok && rp.ok() && py.size() == y.size();
        for (size_t i = 0; ok && i < y.size(); i++) { ok = y[i] == py[i]; }
        // Size validation fails closed (never reads OOB).
        luisa::vector<float> ty, tsum;
        ok = ok && !ref_qwen_rmsnorm_snapshot(x, w, rows, 0u, 1e-6f, ty, tsum).ok();
        ok = ok && !ref_qwen_rmsnorm_snapshot(luisa::span<const float>(x.data(), x.size() - 1u),
                                              w, rows, dim, 1e-6f, ty, tsum).ok();
        ctx.check(ok, "rmsnorm snapshot oracle");
    }
    // --- swiglu split/fused parity.
    {
        const uint32_t m = 2u, f = 9u;
        luisa::vector<float> proj(m * 2u * f);
        for (auto &v : proj) { v = rng.uniform(-3.0f, 3.0f); }
        // Row 0 of the fused projection is [gate f | up f].
        luisa::vector<float> gate(proj.begin(), proj.begin() + f);
        luisa::vector<float> up(proj.begin() + f, proj.begin() + 2 * f);
        luisa::vector<float> o1, o2;
        const auto r1 = ref_qwen_swiglu(gate, up, o1);
        const auto r2 = ref_qwen_swiglu_fused(proj, m, f, o2);
        bool ok = r1.ok() && r2.ok() && o2.size() == static_cast<size_t>(m) * f;
        // Compare row 0 against the split path; check row 1 independently.
        for (size_t i = 0; ok && i < o1.size(); i++) { ok = close_enough(o1[i], o2[i], 1e-6f, 1e-7f); }
        for (uint32_t i = 0; ok && i < f; i++) {
            const double expect = ref_silu(proj[2u * f + i]) * static_cast<double>(proj[3u * f + i]);
            ok = close_enough(o2[f + i], static_cast<float>(expect), 1e-6f, 1e-7f);
        }
        ctx.check(ok, "swiglu split/fused parity");
    }
    // --- attention prepare + prefill + decode end-to-end oracle chain with
    //     GQA (4 q heads, 2 kv heads), partial rope, ragged cache.
    {
        QwenDslConfig cfg;
        cfg.q_heads = 4u;
        cfg.kv_heads = 2u;
        cfg.attn_head_dim = 64u;
        cfg.rope_dim = 16u;
        const uint32_t hd = cfg.attn_head_dim;
        const uint32_t T = 5u;
        const uint32_t pos0 = 3u;// non-zero history
        const uint32_t capacity = 11u;
        luisa::vector<float> q_proj(T * cfg.attn_q_proj_n()), k_proj(T * cfg.attn_kv_n()), v_proj(T * cfg.attn_kv_n());
        luisa::vector<float> qnw(hd), knw(hd);
        luisa::vector<int32_t> positions(T);
        for (auto &v : q_proj) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : k_proj) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : v_proj) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : qnw) { v = rng.uniform(0.7f, 1.3f); }// pre-added GGUF weights
        for (auto &v : knw) { v = rng.uniform(0.7f, 1.3f); }
        for (uint32_t t = 0; t < T; t++) { positions[t] = static_cast<int32_t>(pos0 + t); }
        // Prefill the history [0, pos0) with random normed+roped K/V directly.
        luisa::vector<float> k_cache(capacity * cfg.attn_kv_n()), v_cache(capacity * cfg.attn_kv_n());
        for (auto &v : k_cache) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : v_cache) { v = rng.uniform(-1.0f, 1.0f); }
        luisa::vector<float> q, gate, k, v;
        const auto prep = ref_qwen_attn_prepare(q_proj, k_proj, v_proj, qnw, knw, positions, T, cfg, pos0,
                                                true, k_cache, v_cache, q, gate, k, v);
        ctx.check(prep.ok(), "attn prepare oracle");
        // RoPE tail passthrough + norm-before-rope spot check on q row 0:
        // recompute independently.
        bool ok = prep.ok();
        {
            double ss = 0.0;
            for (uint32_t i = 0; i < hd; i++) { ss += static_cast<double>(q_proj[i]) * q_proj[i]; }
            const double inv = 1.0 / std::sqrt(ss / hd + cfg.norm_eps);
            // Tail element (>= rope_dim) must equal normed value exactly.
            for (uint32_t i = cfg.rope_dim; ok && i < hd; i++) {
                ok = close_enough(q[i], static_cast<float>(q_proj[i] * inv * qnw[i]), 1e-5f, 1e-6f);
            }
            // Gate is raw.
            for (uint32_t i = 0; ok && i < hd; i++) { ok = gate[i] == q_proj[hd + i]; }
        }
        ctx.check(ok, "attn prepare norm/rope/gate semantics");
        const uint32_t kv_len = pos0 + T;
        luisa::vector<float> attn_out;
        const auto pre = ref_qwen_attn_prefill(q, gate, k_cache, v_cache, T, cfg.q_heads, cfg.kv_heads,
                                               hd, cfg.attn_scale(), pos0, kv_len, attn_out);
        ctx.check(pre.ok(), "attn prefill oracle");
        // Decode at the last position must equal the last prefill row.
        luisa::vector<float> q_last(q.begin() + (T - 1u) * cfg.q_heads * hd, q.end());
        luisa::vector<float> gate_last(gate.begin() + (T - 1u) * cfg.q_heads * hd, gate.end());
        luisa::vector<float> dec_out;
        const auto dec = ref_qwen_attn_decode(q_last, gate_last, k_cache, v_cache, cfg.q_heads, cfg.kv_heads,
                                              hd, cfg.attn_scale(), kv_len, dec_out);
        ok = dec.ok() && dec_out.size() == cfg.q_heads * hd;
        for (size_t i = 0; ok && i < dec_out.size(); i++) {
            ok = close_enough(dec_out[i], attn_out[(T - 1u) * cfg.q_heads * hd + i], 1e-5f, 1e-6f);
        }
        ctx.check(ok, "attn decode == prefill last row");
        // Standalone rope oracle parity with the prepare's q rotation.
        luisa::vector<float> rope_out;
        const auto rr = ref_qwen_rope_partial(q, positions, T, cfg.q_heads, hd, cfg.rope_dim, cfg.rope_theta, rope_out);
        ctx.check(rr.ok(), "rope partial oracle runs");
    }
    // --- GDN chain: conv with history across two chunks, gates, qk norm,
    //     recurrent prefill vs decode step-by-step, gated norm.
    {
        QwenDslConfig cfg;
        cfg.gdn_qk_heads = 2u;
        cfg.gdn_v_heads = 4u;
        cfg.gdn_head_dim = 32u;
        const uint32_t hd = cfg.gdn_head_dim;
        const uint32_t C = cfg.gdn_qkv_n();// 2*2*32 + 4*32 = 256
        const uint32_t K = cfg.conv_kernel;
        const uint32_t T1 = 3u, T2 = 4u;
        luisa::vector<float> src1(T1 * C), src2(T2 * C), weight(K * C);
        luisa::vector<float> ring(C * (K - 1u), 0.0f);
        for (auto &v : src1) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : src2) { v = rng.uniform(-1.0f, 1.0f); }
        for (auto &v : weight) { v = rng.uniform(-0.5f, 0.5f); }
        luisa::vector<float> conv1, conv2;
        const auto c1 = ref_qwen_gdn_conv(src1, weight, ring, T1, C, K, 0u, conv1);
        const auto c2 = ref_qwen_gdn_conv(src2, weight, ring, T2, C, K, T1, conv2);
        ctx.check(c1.ok() && c2.ok(), "gdn conv two-chunk history");
        // Independent full-sequence conv (no chunk boundary) must match.
        luisa::vector<float> src_all(src1);
        src_all.insert(src_all.end(), src2.begin(), src2.end());
        luisa::vector<float> ring_zero(C * (K - 1u), 0.0f), conv_all;
        const auto ca = ref_qwen_gdn_conv(src_all, weight, ring_zero, T1 + T2, C, K, 0u, conv_all);
        bool ok = ca.ok();
        for (size_t i = 0; ok && i < conv1.size(); i++) { ok = close_enough(conv1[i], conv_all[i], 1e-5f, 1e-6f); }
        for (size_t i = 0; ok && i < conv2.size(); i++) {
            ok = close_enough(conv2[i], conv_all[conv1.size() + i], 1e-5f, 1e-6f);
        }
        ctx.check(ok, "gdn conv chunked == full");
        // Gates.
        const uint32_t H = cfg.gdn_v_heads;
        const uint32_t T = T1 + T2;
        luisa::vector<float> a(T * H), b(T * H), ssm_a(H), dt_bias(H);
        for (auto &v : a) { v = rng.uniform(-4.0f, 4.0f); }
        for (auto &v : b) { v = rng.uniform(-4.0f, 4.0f); }
        // GGUF ssm_a = -exp(A_log), A_log = log(U(0.01,16)): strictly negative.
        for (auto &v : ssm_a) { v = -std::exp(static_cast<double>(rng.uniform(-4.6f, 2.77f))); }
        for (auto &v : dt_bias) { v = rng.uniform(-1.0f, 1.0f); }
        luisa::vector<float> beta, g;
        const auto gt = ref_qwen_gdn_gates(a, b, ssm_a, dt_bias, T, H, cfg.softplus_threshold, beta, g);
        ok = gt.ok();
        for (auto gv : g) { ok = ok && gv <= 0.0f; }
        // Independent scalar check of element 0 with the GGUF convention.
        {
            const double av = static_cast<double>(a[0]) + dt_bias[0];
            const double sp = av > cfg.softplus_threshold ? av : std::log1p(std::exp(av));
            ok = ok && close_enough(g[0], static_cast<float>(static_cast<double>(ssm_a[0]) * sp), 1e-5f, 1e-6f);
        }
        ctx.check(ok, "gdn gates (negative ssm_a, g <= 0)");
        // QK norm.
        luisa::vector<float> q, k;
        const auto qk = ref_qwen_gdn_qk_norm(conv_all, T, cfg.gdn_qk_heads, hd, cfg.gdn_v_n(),
                                             cfg.l2_eps, cfg.gdn_q_scale(), q, k);
        ok = qk.ok();
        // Verify Q row 0 norm: ||q|| == q_scale (when ||raw|| > eps).
        {
            double ss = 0.0;
            for (uint32_t i = 0; i < hd; i++) { ss += static_cast<double>(q[i]) * q[i]; }
            ok = ok && close_enough(static_cast<float>(std::sqrt(ss)), cfg.gdn_q_scale(), 1e-4f, 1e-6f);
        }
        ctx.check(ok, "gdn qk l2 norm + q scale");
        // Recurrent prefill (T tokens) vs step-by-step decode (T x 1 token).
        luisa::vector<float> state(static_cast<size_t>(H) * hd * hd);
        for (auto &v : state) { v = rng.uniform(-0.1f, 0.1f); }
        luisa::vector<float> state_copy = state;
        luisa::vector<float> out_full;
        const auto rec = ref_qwen_gdn_recurrent(q, k, conv_all, beta, g, state, T, H, hd, cfg.gdn_qk_heads, out_full);
        ctx.check(rec.ok(), "gdn recurrent prefill oracle");
        // Step-by-step: slice per-token q/k/conv/beta/g.
        luisa::vector<float> out_step(static_cast<size_t>(T) * H * hd);
        bool step_ok = true;
        const uint64_t qk_stride = static_cast<uint64_t>(cfg.gdn_qk_heads) * hd;
        const uint64_t c_stride = static_cast<uint64_t>(C);
        for (uint32_t t = 0; t < T && step_ok; t++) {
            luisa::vector<float> qt(q.begin() + t * qk_stride, q.begin() + (t + 1u) * qk_stride);
            luisa::vector<float> kt(k.begin() + t * qk_stride, k.begin() + (t + 1u) * qk_stride);
            luisa::vector<float> ct(conv_all.begin() + t * c_stride, conv_all.begin() + (t + 1u) * c_stride);
            luisa::vector<float> bt(beta.begin() + t * H, beta.begin() + (t + 1u) * H);
            luisa::vector<float> gt1(g.begin() + t * H, g.begin() + (t + 1u) * H);
            luisa::vector<float> ot;
            const auto r = ref_qwen_gdn_recurrent(qt, kt, ct, bt, gt1, state_copy, 1u, H, hd, cfg.gdn_qk_heads, ot);
            step_ok = r.ok();
            for (uint32_t i = 0; i < H * hd && step_ok; i++) { out_step[static_cast<size_t>(t) * H * hd + i] = ot[i]; }
        }
        ok = step_ok;
        for (size_t i = 0; ok && i < out_full.size(); i++) {
            ok = close_enough(out_full[i], out_step[i], 2e-4f, 2e-5f);
        }
        // Final states must match too.
        for (size_t i = 0; ok && i < state.size(); i++) {
            ok = close_enough(state[i], state_copy[i], 2e-4f, 2e-5f);
        }
        ctx.check(ok, "gdn recurrent prefill == step-by-step decode");
        // Gated output norm: direct weight, norm before gate.
        luisa::vector<float> z(static_cast<size_t>(T) * H * hd), nw(hd);
        for (auto &v : z) { v = rng.uniform(-2.0f, 2.0f); }
        for (auto &v : nw) { v = rng.uniform(0.5f, 1.5f); }
        luisa::vector<float> normed;
        const auto gn = ref_qwen_gdn_gated_norm(out_full, z, nw, T, H, hd, cfg.norm_eps, normed);
        ok = gn.ok();
        // Independent row-0 check.
        {
            double ss = 0.0;
            for (uint32_t i = 0; i < hd; i++) { ss += static_cast<double>(out_full[i]) * out_full[i]; }
            const double inv = 1.0 / std::sqrt(ss / hd + cfg.norm_eps);
            for (uint32_t i = 0; ok && i < hd; i++) {
                const double expect = out_full[i] * inv * nw[i] * ref_silu(z[i]);
                ok = close_enough(normed[i], static_cast<float>(expect), 1e-5f, 1e-6f);
            }
        }
        ctx.check(ok, "gdn gated norm (direct weight, norm before gate)");
    }
    // --- planning: overflow refusal, empty plans, flattening.
    {
        const auto empty = plan_qwen_threads(0u);
        ctx.check(empty.ok() && empty.value->empty(), "plan empty");
        const auto big = plan_qwen_threads(0x100000000ull);
        ctx.check(!big.ok(), "plan refuses > uint32 count");
        const auto flat = plan_qwen_threads(static_cast<uint64_t>(kQwenMaxGroupsPerDim) * 256u + 1u);
        ctx.check(flat.ok() && flat.value->is_2d(), "plan flattens past group envelope");
        const auto g2 = plan_qwen_threads_2d(3u, 7u);
        ctx.check(g2.ok() && g2.value->padded_cols >= 7u, "plan 2d ragged");
        const auto bad = plan_qwen_gemv_dispatch(5u, 8u, 16u);
        ctx.check(!bad.ok(), "gemv plan refuses M > 4");
        const auto ov = plan_qwen_gemm_dispatch(0x10000ull, 0x10000ull, 0x10000ull);
        ctx.check(!ov.ok(), "gemm plan refuses overflow");
    }
    // --- config validation.
    {
        QwenDslConfig cfg;
        ctx.check(validate_qwen_config(cfg).ok(), "default config valid");
        cfg.q_heads = 15u;// not a multiple of kv_heads=4
        ctx.check(!validate_qwen_config(cfg).ok(), "config rejects bad GQA ratio");
    }
    // --- WY/UT chunk prefill is part of the supported operator surface.
    ctx.check(dsl_ops_chunk_prefill_status().ok(), "chunk-64 WY/UT prefill available");
    ctx.check(qwen_gdn_chunk_shared_bytes() <= 32u * 1024u, "chunk-64 shared budget <= 32 KiB");
    ctx.check(validate_qwen_gdn_chunk_inputs(65u, 4u, 2u, 32u).ok(), "chunk-64 ragged plan accepted");
    ctx.check(!validate_qwen_gdn_chunk_inputs(65u, 3u, 2u, 32u).ok(), "chunk-64 rejects invalid head pairing");

    // ------------------------------------------------------------------
    // EXTERNAL-CONTRACT FIXTURES (GGUF semantics; independent expectations
    // so a shared oracle/kernel bug cannot pass both sides). Each fixture
    // pins one converter/HF convention that the generic kernels must honor.
    // ------------------------------------------------------------------
    // (a) RMS norm with KNOWN pre-added weights: w = [2, 0.5, 1, 1.5],
    //     x = [3, 4, 0, -1] -> y = x * rsqrt(26/4 + eps) * w (direct).
    {
        const luisa::vector<float> x{3.0f, 4.0f, 0.0f, -1.0f};
        const luisa::vector<float> w{2.0f, 0.5f, 1.0f, 1.5f};
        luisa::vector<float> y, sum;
        const auto r = ref_qwen_rmsnorm(x, w, {}, false, 1u, 4u, 1e-6f, y, sum);
        const double inv = 1.0 / std::sqrt(26.0 / 4.0 + 1e-6);
        bool ok = r.ok();
        const float expect[4] = {float(3.0 * inv * 2.0), float(4.0 * inv * 0.5),
                                 0.0f, float(-1.0 * inv * 1.5)};
        for (uint32_t i = 0; ok && i < 4u; i++) { ok = close_enough(y[i], expect[i], 1e-6f, 1e-7f); }
        ctx.check(ok, "fixture rmsnorm pre-added direct weight");
    }
    // (b) Gates with KNOWN negative ssm_a (GGUF -exp(A_log)):
    //     ssm_a = [-2, -0.5], dt_bias = [0.25, -1], a = [1, 30], b = [0, 2].
    //     g0 = -2 * log1p(exp(1.25)); g1 = -0.5 * 29 (softplus threshold 20).
    {
        const luisa::vector<float> a{1.0f, 30.0f}, b{0.0f, 2.0f};
        const luisa::vector<float> ssm_a{-2.0f, -0.5f}, dt{0.25f, -1.0f};
        luisa::vector<float> beta, g;
        const auto r = ref_qwen_gdn_gates(a, b, ssm_a, dt, 1u, 2u, 20.0f, beta, g);
        const double g0 = -2.0 * std::log1p(std::exp(1.25));
        const double g1 = -0.5 * 29.0;
        const double b0 = 0.5, b1 = 1.0 / (1.0 + std::exp(-2.0));
        const bool ok = r.ok() &&
                        close_enough(g[0], float(g0), 1e-6f, 1e-7f) &&
                        close_enough(g[1], float(g1), 1e-6f, 1e-7f) &&
                        close_enough(beta[0], float(b0), 1e-6f, 1e-7f) &&
                        close_enough(beta[1], float(b1), 1e-6f, 1e-7f);
        ctx.check(ok, "fixture gdn gates negative ssm_a direct");
    }
    // (c) Channel-major impulse conv: C=2, K=4, weight[c*K+j] = c*10+j+1.
    //     src = impulse at t=0 on channel 0 only. Causal depthwise conv (tap
    //     j reads position p-(K-1)+j, i.e. HF F.conv1d with K-1 left padding
    //     truncated to T): the impulse at position 0 is seen by output t
    //     through tap j = K-1-t, so out[t,0] = silu(w[0*K + (K-1-t)]) for
    //     t in [0,K), zero afterwards; channel 1 stays zero. A kernel-major
    //     misread would use w[(K-1-t)*C + 0] and fail. Ring history zero.
    {
        const uint32_t C = 2u, K = 4u, T = 4u;
        luisa::vector<float> w(C * K), src(T * C, 0.0f), ring(C * (K - 1u), 0.0f);
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t j = 0; j < K; j++) { w[c * K + j] = float(c * 10 + j + 1); }
        }
        src[0u * C + 0u] = 1.0f;// impulse: token 0, channel 0
        luisa::vector<float> out;
        const auto r = ref_qwen_gdn_conv(src, w, ring, T, C, K, 0u, out);
        bool ok = r.ok();
        for (uint32_t t = 0; t < T && ok; t++) {
            for (uint32_t c = 0; c < C && ok; c++) {
                const double acc = (c == 0u && t < K) ? static_cast<double>(w[0u * K + (K - 1u - t)]) : 0.0;
                ok = close_enough(out[t * C + c], float(ref_silu(acc)), 1e-6f, 1e-7f);
            }
        }
        ctx.check(ok, "fixture conv channel-major impulse");
    }
    // (d) Tiled V-head pairing: qk_heads=2, v_heads=4, hd=32. Distinct unit
    //     q/k per QK head, distinct v per V head, beta=1, g=0, zero state.
    //     After one token S = k v^T so o[vh] = (q[qk] . k[qk]) * v[vh];
    //     tiled mapping requires qk = vh % 2 (grouped vh/2 would differ).
    {
        const uint32_t QK = 2u, VH = 4u, HD = 32u;
        luisa::vector<float> q(QK * HD, 0.0f), k(QK * HD, 0.0f);
        q[0] = 1.0f; q[HD + 1] = 1.0f;
        k[0] = 1.0f; k[HD + 1] = 1.0f;
        const uint32_t C = 2u * QK * HD + VH * HD;
        luisa::vector<float> conv(C, 0.0f);
        for (uint32_t vh = 0; vh < VH; vh++) { conv[2u * QK * HD + vh * HD + 0u] = float(vh + 1); }
        luisa::vector<float> beta(VH, 1.0f), g(VH, 0.0f), state(VH * HD * HD, 0.0f), out;
        const auto r = ref_qwen_gdn_recurrent(q, k, conv, beta, g, state, 1u, VH, HD, QK, out);
        bool ok = r.ok();
        for (uint32_t vh = 0; vh < VH && ok; vh++) {
            ok = close_enough(out[vh * HD + 0u], float(vh + 1), 1e-6f, 1e-7f);
            for (uint32_t i = 1; i < HD && ok; i++) { ok = out[vh * HD + i] == 0.0f; }
        }
        ctx.check(ok, "fixture gdn tiled v-head pairing (h % qk_heads)");
    }
    // (e) NeoX rotate-half rope: hd=8, rope_dim=4 (half=2), theta=10000,
    //     pos=1, x=[1,2,3,4,5,6,7,8]. freq[p]=10000^(-2p/4).
    //     out[0]=c0-3s0; out[2]=s0+3c0; out[1]=2c1-4s1; out[3]=2s1+4c1;
    //     tail [4..8) passthrough.
    {
        const luisa::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
        const luisa::vector<int32_t> pos{1};
        luisa::vector<float> out;
        const auto r = ref_qwen_rope_partial(x, pos, 1u, 1u, 8u, 4u, 10000.0f, out);
        const double f0 = 1.0, f1 = std::pow(10000.0, -0.5);
        const double c0 = std::cos(f0), s0 = std::sin(f0);
        const double c1 = std::cos(f1), s1 = std::sin(f1);
        const float expect[8] = {float(1 * c0 - 3 * s0), float(2 * c1 - 4 * s1),
                                 float(1 * s0 + 3 * c0), float(2 * s1 + 4 * c1),
                                 5, 6, 7, 8};
        bool ok = r.ok();
        for (uint32_t i = 0; ok && i < 8u; i++) { ok = close_enough(out[i], expect[i], 1e-5f, 1e-6f); }
        ctx.check(ok, "fixture rope NeoX rotate-half + tail");
    }
    // (f) Near-zero L2 rows: conv row q = [1e-4, 0...], k = [0...].
    //     FLA l2norm: q_out = x * rsqrt(sum+eps) * q_scale (tiny, NOT ~1/eps
    //     blown up); k_out exactly 0.
    {
        const uint32_t QK = 1u, HD = 32u, VN = 32u;
        const uint32_t C = 2u * QK * HD + VN;
        luisa::vector<float> conv(C, 0.0f);
        conv[0] = 1e-4f;
        luisa::vector<float> q, k;
        const float qs = 1.0f / std::sqrt(32.0f);
        const auto r = ref_qwen_gdn_qk_norm(conv, 1u, QK, HD, VN, 1e-6f, qs, q, k);
        const double inv = 1.0 / std::sqrt(1e-8 + 1e-6);
        bool ok = r.ok() && close_enough(q[0], float(1e-4 * inv * qs), 1e-5f, 1e-9f);
        for (uint32_t i = 1; i < HD && ok; i++) { ok = q[i] == 0.0f; }
        for (uint32_t i = 0; i < HD && ok; i++) { ok = k[i] == 0.0f; }
          ctx.check(ok, "fixture gdn l2norm near-zero rows (eps inside rsqrt)");
      }

      // --- weight-reusing prefill GEMM (opt_plan/02 section 3): host contract.
      {
          const QwenDslConfig model{};
          QwenGemmMtConfig cfg{};
          bool ok = validate_qwen35_gemm_mt_config(cfg).ok() && cfg.mt == 16u && cfg.warps == 4u;
          QwenGemmMtConfig bad{};
          bad.mt = 0u;
          ok = ok && !validate_qwen35_gemm_mt_config(bad).ok();
          bad.mt = 33u;
          ok = ok && !validate_qwen35_gemm_mt_config(bad).ok();
          bad.mt = 16u;
          bad.warps = 0u;
          ok = ok && !validate_qwen35_gemm_mt_config(bad).ok();
          bad.warps = 9u;
          ok = ok && !validate_qwen35_gemm_mt_config(bad).ok();
          QwenGemmMtConfig selected{};
          ok = ok && select_qwen35_gemm_mt_config(32u, selected).ok() && selected.mt == 16u;
          QwenGemmMtConfig rejected = cfg;
          ok = ok && !select_qwen35_gemm_mt_config(64u, rejected).ok();
          ctx.check(ok, "mt gemm config validation + warp-32 selection");
      }
      {
          QwenGemmMtConfig cfg{};
          const QwenDslConfig model{};
          // Model shape: N=9216, K=2560, M=65 -> 2304 column blocks x 5 stripes.
          auto plan = plan_qwen35_gemm_mt_dispatch(65u, 9216u, 2560u, QwenWeightFormat::bf16, 64u, cfg);
          bool ok = plan.ok();
          if (ok) {
              const auto &p = *plan.value;
              ok = p.mt == 16u && p.columns_per_block == 4u && p.column_blocks == 2304u &&
                   p.stripes == 5u && p.blocks == 11520u && p.total_x == 2304u * 128u &&
                   p.total_y == 5u;
          }
          ctx.check(ok, "mt gemm plan: one 128-thread block per (4 columns, mt rows)");
          // Ragged shapes: M=1 is one stripe, N=32 is 8 column blocks, N=37 is 10.
          auto m1 = plan_qwen35_gemm_mt_dispatch(1u, 32u, 2560u, QwenWeightFormat::bf16, 64u, cfg);
          ok = m1.ok() && m1.value->stripes == 1u && m1.value->column_blocks == 8u &&
               m1.value->blocks == 8u;
          auto m17 = plan_qwen35_gemm_mt_dispatch(17u, 37u, 2560u, QwenWeightFormat::fp8_e4m3, 64u, cfg);
          ok = ok && m17.ok() && m17.value->stripes == 2u && m17.value->column_blocks == 10u &&
               m17.value->blocks == 20u;
          // Empty work yields an empty plan (the executor skips the dispatch).
          auto empty = plan_qwen35_gemm_mt_dispatch(0u, 37u, 2560u, QwenWeightFormat::bf16, 64u, cfg);
          ok = ok && empty.ok() && empty.value->empty();
          ctx.check(ok, "mt gemm plan: ragged M/N, empty work");
          // Packed-word contract: K must be a multiple of the pack width and the
          // fp8 group a multiple of 4, otherwise the reference GEMM is required.
          ok = !plan_qwen35_gemm_mt_dispatch(8u, 32u, 129u, QwenWeightFormat::bf16, 64u, cfg).ok();
          ok = ok && !plan_qwen35_gemm_mt_dispatch(8u, 32u, 130u, QwenWeightFormat::fp8_e4m3, 64u, cfg).ok();
          ok = ok && !plan_qwen35_gemm_mt_dispatch(8u, 32u, 2560u, QwenWeightFormat::fp8_e4m3, 62u, cfg).ok();
          ok = ok && !plan_qwen35_gemm_mt_dispatch(8u, 32u, 2560u, QwenWeightFormat::fp8_e4m3, 0u, cfg).ok();
          ok = ok && plan_qwen35_gemm_mt_dispatch(8u, 32u, 128u, QwenWeightFormat::fp8_e4m3, 64u, cfg).ok();
          ok = ok && plan_qwen35_gemm_mt_dispatch(8u, 32u, 2560u, QwenWeightFormat::fp8_e4m3, 64u, cfg).ok();
          ok = ok && plan_qwen35_gemm_mt_dispatch(8u, 32u, 2560u, QwenWeightFormat::bf16, 64u, cfg).ok();
          ctx.check(ok, "mt gemm plan: packed-word/group contract fails closed");
          // Portable grid envelope: N wider than 65535 column blocks (i.e.
          // N > 262140) fails closed instead of launching a rejected grid.
          ok = plan_qwen35_gemm_mt_dispatch(64u, 262140u, 2560u, QwenWeightFormat::bf16, 64u, cfg).ok();
          ok = ok && !plan_qwen35_gemm_mt_dispatch(64u, 262144u, 2560u, QwenWeightFormat::bf16, 64u, cfg).ok();
          ctx.check(ok, "mt gemm plan: 65535-groups-per-dimension envelope");
          (void)model;
      }

      // --- activation workspace geometry / aliasing plan (09 V6 + N3.5).
      {
          const QwenDslConfig model{};
          const auto geo = qwen_workspace_geometry(model, 1u);
          bool ok = geo.arena_floats[0] == model.ffn && geo.arena_floats[1] == model.ffn &&
                    geo.arena_floats[2] == model.ffn &&
                    geo.arena_floats[3] == model.gdn_v_n() &&
                    geo.arena_floats[4] == model.gdn_v_n() &&
                    geo.arena_floats[5] == model.gdn_v_n() &&
                    geo.arena_floats[6] == model.gdn_qk_n() &&
                    geo.arena_floats[7] == model.gdn_qk_n() &&
                    geo.arena_floats[8] == model.gdn_v_heads &&
                    geo.arena_floats[9] == model.gdn_v_heads;
          // 3 x ffn + 3 x gdn_v + 2 x gdn_qk + 2 x gdn_gate per chunk row.
          ok = ok && geo.pool_floats_per_row == 3u * model.ffn + 3u * model.gdn_v_n() +
                                                    2u * model.gdn_qk_n() + 2u * model.gdn_v_heads;
          ok = ok && geo.hidden_elems == model.hidden && geo.normed_elems == model.hidden &&
                    geo.residual_elems == model.hidden;
          ok = ok && geo.chunk_bytes ==
                         (geo.pool_floats_per_row + 3u * model.hidden) * sizeof(float);
          ctx.check(ok, "workspace geometry: 10 arenas, sizes from the model config");
          // Row scaling + the honest before/after ratio: the pre-aliasing
          // executor allocated 116320 floats per chunk row for this model.
          const uint64_t legacy_floats_per_row = 116320u;
          const auto geo1024 = qwen_workspace_geometry(model, 1024u);
          ok = geo1024.pool_elems == geo1024.pool_floats_per_row * 1024u &&
               geo1024.chunk_bytes == geo1024.chunk_elems * sizeof(float) &&
               geo1024.pool_floats_per_row + 3u * model.hidden < legacy_floats_per_row / 2u;
          ok = ok && geo1024.chunk_bytes < 210u * 1024u * 1024u;
          ctx.check(ok, "workspace geometry: chunk 1024 stays under 210 MiB (was 454 MiB)");
      }
      {
          // Lifetime proof: every pair of members sharing an arena must be
          // disjoint in the recorded order, except a GDN-only vs a
          // full-attention-only pair (a layer is exactly one of the two).
          const QwenDslConfig model{};
          const auto members = qwen_workspace_members(model);
          bool ok = members.size() == 27u;
          for (size_t i = 0; i < members.size(); i++) {
              const auto &a = members[i];
              if (a.arena_index >= kQwenWorkspaceArenaCount) {
                  // Permanent members are single-owner by construction.
                  ok = ok && a.stage == QwenWorkspaceStage::permanent;
                  continue;
              }
              for (size_t j = i + 1u; j < members.size(); j++) {
                  const auto &b = members[j];
                  if (b.arena_index != a.arena_index) { continue; }
                  const bool exclusive_layers =
                      (a.stage == QwenWorkspaceStage::gdn && b.stage == QwenWorkspaceStage::attention) ||
                      (a.stage == QwenWorkspaceStage::attention && b.stage == QwenWorkspaceStage::gdn);
                  const bool disjoint = a.last_step < b.first_step || b.last_step < a.first_step;
                  if (!exclusive_layers && !disjoint) { ok = false; }
              }
          }
          // The arena size must be the maximum of its members.
          for (uint32_t arena = 0u; arena < kQwenWorkspaceArenaCount; arena++) {
              uint64_t want = 0u;
              for (const auto &m : members) {
                  if (m.arena_index == arena) { want = std::max(want, m.floats_per_row); }
              }
              ok = ok && want != 0u && qwen_workspace_geometry(model, 1u).arena_floats[arena] == want;
          }
          ctx.check(ok, "workspace lifetime: aliased members are provably disjoint");
      }

      // --- adaptive prefill chunk rule (opt_plan/02 section 4).
      {
          bool ok = qwen_next_pow2_u32(0u) == 1u && qwen_next_pow2_u32(1u) == 1u &&
                    qwen_next_pow2_u32(2u) == 2u && qwen_next_pow2_u32(3u) == 4u &&
                    qwen_next_pow2_u32(5u) == 8u && qwen_next_pow2_u32(1024u) == 1024u &&
                    qwen_next_pow2_u32(1025u) == 2048u;
          ok = ok && qwen_next_pow2_u32(0x1FFFFFFFFull) == 0xFFFFFFFFu;
          ctx.check(ok, "next_pow2 clamping");
          const auto p1 = qwen_plan_prefill_chunks(1u, 1024u, 1024u);
          ok = p1.chunk == 1u && p1.chunks == 1u;
          const auto p64 = qwen_plan_prefill_chunks(64u, 1024u, 1024u);
          ok = ok && p64.chunk == 64u && p64.chunks == 1u;
          const auto p128 = qwen_plan_prefill_chunks(128u, 1024u, 1024u);
          ok = ok && p128.chunk == 64u && p128.chunks == 2u;
          const auto p512 = qwen_plan_prefill_chunks(512u, 1024u, 1024u);
          ok = ok && p512.chunk == 128u && p512.chunks == 4u;
          const auto p1024 = qwen_plan_prefill_chunks(1024u, 1024u, 1024u);
          ok = ok && p1024.chunk == 256u && p1024.chunks == 4u;
          const auto p4096 = qwen_plan_prefill_chunks(4096u, 1024u, 1024u);
          ok = ok && p4096.chunk == 1024u && p4096.chunks == 4u;
          ctx.check(ok, "adaptive chunk: clamp(next_pow2(S/4),64,max) for S=1..4096");
          // Ragged last chunk: 1000 rows in chunks of 256 -> 4 chunks (last 232).
          const auto ragged = qwen_plan_prefill_chunks(1000u, 1024u, 1024u);
          bool ok2 = ragged.chunk == 256u && ragged.chunks == 4u &&
                     (ragged.chunks - 1u) * ragged.chunk + 232u == 1000u;
          // Configured max below the rule, and the per-backend conservative cap.
          const auto small = qwen_plan_prefill_chunks(4096u, 256u, 1024u);
          ok2 = ok2 && small.chunk == 256u && small.chunks == 16u && !small.limited_by_backend;
          const auto capped = qwen_plan_prefill_chunks(4096u, 1024u, 512u);
          ok2 = ok2 && capped.chunk == 512u && capped.chunks == 8u && capped.limited_by_backend &&
                 capped.max_chunk == 512u;
          // A tiny backend limit degrades to that limit instead of 0.
          const auto tiny_limit = qwen_plan_prefill_chunks(4096u, 1024u, 16u);
          ok2 = ok2 && tiny_limit.chunk == 16u && tiny_limit.chunks == 256u;
          ok2 = ok2 && qwen_prefill_chunk_backend_limit("cuda") == 1024u &&
                 qwen_prefill_chunk_backend_limit("vk") == 1024u &&
                 qwen_prefill_chunk_backend_limit("dx") == 512u &&
                 qwen_prefill_chunk_backend_limit("metal") == 512u;
          const auto zero_rows = qwen_plan_prefill_chunks(0u, 1024u, 1024u);
          ok2 = ok2 && zero_rows.chunk >= 1u && zero_rows.chunks == 0u;
          ctx.check(ok2, "adaptive chunk: ragged splits, max clamp, backend cap");
      }

      if (ctx.failures != 0u) {

        return Status::error(StatusCode::invalid_argument,
                                luisa::format("dsl_ops_self_test: {} check(s) failed", ctx.failures));
    }
    return Status::success();
}

}// namespace luisa::qwen::dsl
