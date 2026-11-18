// Qwen3.5-4B shared engine layer -- common host contract implementation.
// Bit-exact software BF16/FP8 codecs and checked packing. No exceptions.
#include "qwen/common.h"

#include <cmath>

namespace luisa::qwen {

[[nodiscard]] bool checked_element_count(luisa::span<const uint64_t> dims, uint64_t &out) noexcept {
    uint64_t count = 1;
    for (auto d : dims) {
        if (!checked_mul(count, d, count)) { return false; }
    }
    out = count;
    return true;
}

// ---------------------------------------------------------------------------
// BF16
// ---------------------------------------------------------------------------
[[nodiscard]] float bf16_to_f32(uint16_t b) noexcept {
    const uint32_t bits = static_cast<uint32_t>(b) << 16u;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

[[nodiscard]] uint16_t f32_to_bf16(float f) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t exp = bits & 0x7f800000u;
    const uint32_t mant = bits & 0x007fffffu;
    if (exp == 0x7f800000u && mant != 0u) {
        // NaN -> quiet BF16 NaN, keep the sign and a nonzero payload.
        return static_cast<uint16_t>((bits >> 16u) | 0x0040u);
    }
    // Round to nearest even on the truncated low 16 bits. This also carries
    // overflow into the exponent correctly (finite overflow -> +/-inf).
    const uint32_t lsb = (bits >> 16u) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16u);
}

// ---------------------------------------------------------------------------
// FP8 E4M3FN
// ---------------------------------------------------------------------------
[[nodiscard]] float fp8_e4m3_to_f32(uint8_t code) noexcept {
    const uint32_t sign = static_cast<uint32_t>(code & 0x80u) << 24u;
    const uint32_t exp = (code >> 3u) & 0x0fu;
    const uint32_t mant = code & 0x07u;
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;// +/-0
        } else {
            // Subnormal: value = mant * 2^-9. Normalize into an FP32 normal:
            // shift the 3-bit mantissa left until bit 3 (the implicit 1) is
            // set, tracking the exponent. mant=1 -> 2^-9, mant=4 -> 2^-7.
            uint32_t m = mant;
            int e = -6;// value = m * 2^-9 with m in [1,7]; normalize m to [8,15]
            while ((m & 0x8u) == 0u) {
                m <<= 1u;
                --e;
            }
            m &= 0x7u;
            bits = sign | (static_cast<uint32_t>(e + 127) << 23u) | (m << 20u);
        }
    } else if (exp == 0x0fu && mant == 0x07u) {
        bits = sign | 0x7fc00000u;// quiet NaN
    } else {
        // Normal: exponent bias 7 -> 127, mantissa 3 -> 23 bits.
        bits = sign | ((exp + 120u) << 23u) | (mant << 20u);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

[[nodiscard]] uint8_t f32_to_fp8_e4m3(float f) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint8_t sign = static_cast<uint8_t>((bits >> 24u) & 0x80u);
    const uint32_t exp = (bits >> 23u) & 0xffu;
    const uint32_t mant = bits & 0x007fffffu;
    if (exp == 0xffu) {
        if (mant != 0u) { return kFp8E4M3Nan; }
        // Infinity: saturate to the largest finite magnitude.
        return static_cast<uint8_t>(sign | 0x7eu);
    }
    if (exp == 0u && mant == 0u) { return sign; }// +/-0
    // Exact decimal value as an integer multiple of 2^-9 (the E4M3 subnormal
    // step). FP32 has 24 significand bits; shifting stays inside uint64_t.
    int e = static_cast<int>(exp) - 127;
    uint64_t sig;
    if (exp == 0u) {
        sig = mant;
        e = -126;
    } else {
        sig = mant | 0x00800000u;
    }
    // value = sig * 2^(e-23); scaled = value / 2^-9 = sig * 2^(e-14).
    const int shift = e - 14;
    uint64_t scaled;
    if (shift >= 0) {
        if (shift > 40) { return static_cast<uint8_t>(sign | 0x7eu); }// huge -> saturate
        scaled = sig << static_cast<unsigned>(shift);
    } else {
        const unsigned r = static_cast<unsigned>(-shift);
        // RNE on the discarded bits, with a sticky bit so shifts >= 64 stay
        // exact: q = 0, rem = sig, half is "any value >= 2^(r-1)". For r >= 64
        // sig < 2^24 <= half, and sig == half is impossible, so the result is
        // 0 -- except the exact tie at r == 24 (value == 2^-10), where RNE
        // rounds to the even neighbor 0 as well.
        const uint64_t q = r >= 64u ? 0ull : (sig >> r);
        const uint64_t rem = r >= 64u ? sig : (sig & ((1ull << r) - 1ull));
        const bool gt_half = r >= 64u ? false : rem > (1ull << (r - 1u));
        const bool eq_half = r >= 1u && r <= 63u && rem == (1ull << (r - 1u));
        scaled = q + ((gt_half || (eq_half && (q & 1u) != 0u)) ? 1u : 0u);
    }
    // scaled is the magnitude in units of 2^-9. Max finite = 448 * 512 =
    // 229376 = 0x38000. Saturate anything that would round past 448.
    if (scaled >= 0x38000ull) { return static_cast<uint8_t>(sign | 0x7eu); }
    if (scaled < 8ull) {
        // Subnormal (or rounded up into the smallest normal, code 0x08).
        return static_cast<uint8_t>(sign | static_cast<uint8_t>(scaled));
    }
    // Normal: value = q * 2^(p-3) * 2^-9 with q in [8,15] (4 significand
    // bits). The unbiased E4M3 exponent is (p - 3) - 9 = p - 12, so the
    // biased field is p - 12 + 7 = p - 5. (p == 12 -> field 7 -> 1.0.)
    // NOTE: scaled is in units of 2^-9, so value = scaled * 2^-9 and the
    // leading-bit position p satisfies 2^p <= scaled < 2^(p+1), i.e. the
    // value's exponent is p - 9, mantissa q = scaled >> (p-3) in [8,15].
    // Biased field = (p - 9) + 7 = p - 2.
    unsigned p = 63u;
    while ((scaled & (1ull << p)) == 0ull) { --p; }
    const unsigned r = p - 3u;
    uint64_t q = scaled >> r;
    const uint64_t rem = scaled & ((1ull << r) - 1ull);
    // r == 0 means scaled is already a 4-bit significand: there are no
    // discarded bits and no rounding half to compute. Guard the shift rather
    // than allowing r-1 to wrap (which would be undefined behaviour).
    const uint64_t half = r == 0u ? 0ull : (1ull << (r - 1u));
    q += (rem > half || (rem == half && r != 0u && (q & 1u) != 0u)) ? 1u : 0u;
    if (q == 16ull) {
        // Mantissa carried into the next exponent.
        q = 8ull;
        ++p;
    }
    const uint32_t exp_field = p - 2u;
    if (exp_field > 0x0fu || (exp_field == 0x0fu && (q & 0x7u) == 0x7u)) {
        return static_cast<uint8_t>(sign | 0x7eu);// overflow / NaN code -> saturate
    }
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(exp_field << 3u) |
                                static_cast<uint8_t>(q & 0x7u));
}

// ---------------------------------------------------------------------------
// Packing helpers
// ---------------------------------------------------------------------------
[[nodiscard]] Status pack_bf16_pairs(luisa::span<const uint16_t> codes,
                                     luisa::span<uint32_t> packed_out) noexcept {
    const uint64_t words = (static_cast<uint64_t>(codes.size()) + 1u) / 2u;
    if (static_cast<uint64_t>(packed_out.size()) != words) {
        return Status::error(StatusCode::invalid_argument,
                             "bf16 pack output size mismatch", "packed_out");
    }
    for (uint64_t w = 0; w < words; ++w) {
        const uint32_t lo = codes[2u * w];
        const uint32_t hi = (2u * w + 1u < codes.size()) ? codes[2u * w + 1u] : 0u;
        packed_out[w] = lo | (hi << 16u);
    }
    return Status::success();
}

[[nodiscard]] Status unpack_bf16_pairs(luisa::span<const uint32_t> packed, uint64_t count,
                                       luisa::span<uint16_t> codes_out) noexcept {
    if (static_cast<uint64_t>(codes_out.size()) != count) {
        return Status::error(StatusCode::invalid_argument,
                             "bf16 unpack output size mismatch", "codes_out");
    }
    const uint64_t words = (count + 1u) / 2u;
    if (static_cast<uint64_t>(packed.size()) != words) {
        return Status::error(StatusCode::invalid_argument,
                             "bf16 unpack input size mismatch", "packed");
    }
    for (uint64_t i = 0; i < count; ++i) {
        const uint32_t word = packed[i / 2u];
        codes_out[i] = static_cast<uint16_t>((word >> (16u * (i % 2u))) & 0xffffu);
    }
    if ((count % 2u) != 0u && (packed[words - 1u] >> 16u) != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "bf16 unpack padding half is nonzero", "packed");
    }
    return Status::success();
}

[[nodiscard]] Status pack_fp8_quads(luisa::span<const uint8_t> codes,
                                    luisa::span<uint32_t> packed_out) noexcept {
    const uint64_t words = (static_cast<uint64_t>(codes.size()) + 3u) / 4u;
    if (static_cast<uint64_t>(packed_out.size()) != words) {
        return Status::error(StatusCode::invalid_argument,
                             "fp8 pack output size mismatch", "packed_out");
    }
    for (uint64_t w = 0; w < words; ++w) {
        uint32_t word = 0;
        for (uint32_t b = 0; b < 4u; ++b) {
            const uint64_t i = 4u * w + b;
            if (i < codes.size()) { word |= static_cast<uint32_t>(codes[i]) << (8u * b); }
        }
        packed_out[w] = word;
    }
    return Status::success();
}

[[nodiscard]] Status unpack_fp8_quads(luisa::span<const uint32_t> packed, uint64_t count,
                                      luisa::span<uint8_t> codes_out) noexcept {
    if (static_cast<uint64_t>(codes_out.size()) != count) {
        return Status::error(StatusCode::invalid_argument,
                             "fp8 unpack output size mismatch", "codes_out");
    }
    const uint64_t words = (count + 3u) / 4u;
    if (static_cast<uint64_t>(packed.size()) != words) {
        return Status::error(StatusCode::invalid_argument,
                             "fp8 unpack input size mismatch", "packed");
    }
    for (uint64_t i = 0; i < count; ++i) {
        codes_out[i] = static_cast<uint8_t>((packed[i / 4u] >> (8u * (i % 4u))) & 0xffu);
    }
    if ((count % 4u) != 0u) {
        const uint32_t last = packed[words - 1u];
        for (uint32_t b = static_cast<uint32_t>(count % 4u); b < 4u; ++b) {
            if (((last >> (8u * b)) & 0xffu) != 0u) {
                return Status::error(StatusCode::invalid_argument,
                                     "fp8 unpack padding byte is nonzero", "packed");
            }
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Storage classification
// ---------------------------------------------------------------------------
[[nodiscard]] luisa::string_view weight_storage_name(WeightStorage s) noexcept {
    switch (s) {
        case WeightStorage::f32_dense: return "f32_dense";
        case WeightStorage::bf16_packed: return "bf16_packed";
        case WeightStorage::fp8_packed: return "fp8_packed";
        default: return "unknown";
    }
}

[[nodiscard]] Result<WeightStorage> classify_weight_storage(
    luisa::string_view name, uint32_t gguf_type,
    luisa::span<const uint64_t> dims, bool want_fp8) noexcept {
    // GGUF tensor types: F32 = 0, Q8_0 = 8, BF16 = 30 (see gguf.h).
    constexpr uint32_t kGgufF32 = 0u;
    constexpr uint32_t kGgufQ8_0 = 8u;
    constexpr uint32_t kGgufBF16 = 30u;
    if (gguf_type == kGgufF32) { return Result<WeightStorage>::success(WeightStorage::f32_dense); }
    if (gguf_type == kGgufQ8_0) {
        return Result<WeightStorage>::failure(
            StatusCode::unsupported,
            "Q8_0 is INT8 block quantization, not FP8; explicit dequantization "
            "and re-quantization is required (double quantization must be reported)",
            luisa::string{name});
    }
    if (gguf_type != kGgufBF16) {
        return Result<WeightStorage>::failure(
            StatusCode::unsupported, "unsupported GGUF tensor type for weights",
            luisa::string{name});
    }
    // Dense-sensitive tensors: 1-D (norms / ssm_a / ssm_dt.bias) and the
    // depthwise convolution always stay F32 regardless of stored dtype.
    const bool conv = name.ends_with(".ssm_conv1d.weight");
    if (dims.size() < 2u || conv) {
        return Result<WeightStorage>::success(WeightStorage::f32_dense);
    }
    return Result<WeightStorage>::success(want_fp8 ? WeightStorage::fp8_packed
                                                   : WeightStorage::bf16_packed);
}

}// namespace luisa::qwen
