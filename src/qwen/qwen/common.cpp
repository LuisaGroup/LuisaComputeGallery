// Qwen3.5-4B shared engine layer -- common host contract implementation.
// Bit-exact software BF16/FP8 codecs and checked packing. No exceptions.
#include "qwen/common.h"

#include <cmath>
#include <limits>

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
// FP16 (Q8_0 block scale)
// ---------------------------------------------------------------------------
[[nodiscard]] float fp16_to_f32(uint16_t bits) noexcept {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16u;
    const uint32_t exp = (bits >> 10u) & 0x1fu;
    const uint32_t mant = bits & 0x3ffu;
    uint32_t out;
    if (exp == 0u) {
        if (mant == 0u) {
            out = sign;// +/-0
        } else {
            // Subnormal: value = mant * 2^-24. Normalize into an FP32 normal.
            uint32_t m = mant;
            int e = 0;// value = m * 2^-24 with m in [1,1023]
            while ((m & 0x400u) == 0u) {
                m <<= 1u;
                --e;
            }
            m &= 0x3ffu;
            // value = (1024 + m) * 2^(-24 + e) = (1 + m/1024) * 2^(e - 14), so
            // the FP32 exponent field is (e - 14) + 127 = e + 113.
            out = sign | (static_cast<uint32_t>(e + 113) << 23u) | (m << 13u);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (mant << 13u);// inf / NaN (payload preserved)
    } else {
        out = sign | ((exp + 112u) << 23u) | (mant << 13u);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

[[nodiscard]] uint16_t f32_to_fp16(float f) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    const uint32_t exp = (bits >> 23u) & 0xffu;
    const uint32_t mant = bits & 0x007fffffu;
    if (exp == 0xffu) {
        if (mant != 0u) { return static_cast<uint16_t>(sign | 0x7e00u); }// quiet NaN
        return static_cast<uint16_t>(sign | 0x7c00u);                    // inf
    }
    const int unbiased = static_cast<int>(exp) - 127;
    if (unbiased > 15) { return static_cast<uint16_t>(sign | 0x7c00u); }// overflow -> inf
    if (unbiased >= -14) {
        // Normal FP16. Round the 13 discarded mantissa bits RNE.
        uint32_t m = mant >> 13u;
        const uint32_t rem = mant & 0x1fffu;
        const uint32_t half = 0x1000u;
        if (rem > half || (rem == half && (m & 1u) != 0u)) {
            ++m;
            if (m == 0x400u) {
                // Carry into the exponent (the leading bit overflowed).
                m = 0u;
                const uint32_t e = static_cast<uint32_t>(unbiased + 1) + 15u;
                if (e >= 0x1fu) { return static_cast<uint16_t>(sign | 0x7c00u); }
                return static_cast<uint16_t>(sign | (e << 10u));
            }
        }
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(unbiased + 15) << 10u) | m);
    }
    // Subnormal/underflow: value = mant2 * 2^-24. Round RNE with a sticky path.
    const int shift = -unbiased - 1;// total right shift of the 24-bit significand
    const uint32_t sig = mant | 0x00800000u;
    if (shift >= 32) { return static_cast<uint16_t>(sign); }
    const uint32_t q = sig >> static_cast<unsigned>(shift);
    const uint32_t rem = sig & ((1u << static_cast<unsigned>(shift)) - 1u);
    const uint32_t half = 1u << static_cast<unsigned>(shift - 1);
    uint32_t m = q;
    if (rem > half || (rem == half && (q & 1u) != 0u)) { ++m; }
    // m may reach 0x400 (smallest normal), which is a valid FP16 encoding.
    return static_cast<uint16_t>(sign | m);
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
        case WeightStorage::q8_0_packed: return "q8_0_packed";
        case WeightStorage::w4_packed: return "w4_packed";
        default: return "unknown";
    }
}

[[nodiscard]] Result<WeightStorage> classify_weight_storage(
    luisa::string_view name, uint32_t gguf_type,
    luisa::span<const uint64_t> dims, bool want_fp8, bool want_q8_0) noexcept {
    // GGUF tensor types: F32 = 0, Q8_0 = 8, BF16 = 30 (see gguf.h).
    constexpr uint32_t kGgufF32 = 0u;
    constexpr uint32_t kGgufQ8_0 = 8u;
    constexpr uint32_t kGgufBF16 = 30u;
    if (want_fp8 && want_q8_0) {
        return Result<WeightStorage>::failure(
            StatusCode::invalid_argument,
            "FP8 and native Q8_0 weight storage are mutually exclusive (Q8_0 is never "
            "re-quantized into FP8 and FP8 is never substituted for Q8_0)",
            luisa::string{name});
    }
    if (gguf_type == kGgufF32) { return Result<WeightStorage>::success(WeightStorage::f32_dense); }
    if (gguf_type == kGgufQ8_0) {
        if (!want_q8_0) {
            return Result<WeightStorage>::failure(
                StatusCode::unsupported,
                "Q8_0 is INT8 block quantization, not FP8; the native Q8_0 device format is "
                "opt-in (dsl-q8 / QwenWeightUploadOptions::q8_0) and explicit dequantization + "
                "re-quantization would have to be reported as double quantization",
                luisa::string{name});
        }
        // Sensitive tensors (1-D norms/ssm_a/ssm_dt.bias and the depthwise
        // convolution) keep dense F32 storage; a Q8_0 source for them is a
        // checkpoint we do not accept (we never silently dequantize it).
        const bool conv = name.ends_with(".ssm_conv1d.weight");
        if (dims.size() < 2u || conv) {
            return Result<WeightStorage>::failure(
                StatusCode::unsupported,
                "Q8_0 is only supported for packed matrix weights; sensitive tensors "
                "(norms, ssm_a, ssm_dt.bias, ssm_conv1d) must stay F32",
                luisa::string{name});
        }
        if (dims[0] == 0u || dims[0] % kQ8_0BlockSize != 0u) {
            return Result<WeightStorage>::failure(
                StatusCode::unsupported,
                "Q8_0 requires K % 32 == 0 (whole 32-weight blocks and whole 8-word groups)",
                luisa::string{name});
        }
        return Result<WeightStorage>::success(WeightStorage::q8_0_packed);
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

// ---------------------------------------------------------------------------
// Q8_0 storage ABI
// ---------------------------------------------------------------------------
[[nodiscard]] Status pack_q8_0_blocks(luisa::span<const std::byte> payload,
                                      uint64_t element_count,
                                      luisa::span<uint32_t> words_out,
                                      luisa::span<float> scales_out) noexcept {
    if (element_count == 0u || element_count % kQ8_0BlockSize != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "q8_0 payload element count is not a positive multiple of 32",
                             "element_count");
    }
    const uint64_t blocks = element_count / kQ8_0BlockSize;
    uint64_t expect_bytes = 0;
    if (!checked_mul(blocks, kQ8_0BlockBytes, expect_bytes) ||
        static_cast<uint64_t>(payload.size()) != expect_bytes) {
        return Status::error(StatusCode::invalid_argument,
                             "q8_0 payload byte size mismatch", "payload");
    }
    if (static_cast<uint64_t>(scales_out.size()) != blocks) {
        return Status::error(StatusCode::invalid_argument,
                             "q8_0 scale output size mismatch", "scales_out");
    }
    // One uint32 word per 4 codes; LOW element first (endian-explicit).
    if (static_cast<uint64_t>(words_out.size()) != element_count / 4u) {
        return Status::error(StatusCode::invalid_argument,
                             "q8_0 packed output size mismatch", "words_out");
    }
    for (uint64_t b = 0; b < blocks; ++b) {
        const std::byte *src = payload.data() + b * kQ8_0BlockBytes;
        const auto lo = static_cast<uint32_t>(static_cast<uint8_t>(src[0]));
        const auto hi = static_cast<uint32_t>(static_cast<uint8_t>(src[1]));
        scales_out[b] = fp16_to_f32(static_cast<uint16_t>(lo | (hi << 8u)));
        for (uint64_t w = 0; w < kQ8_0BlockSize / 4u; ++w) {
            const std::byte *codes = src + 2u + w * 4u;
            // Byte copy: the int8 codes are exactly the checkpoint's codes.
            uint32_t word = 0;
            for (uint32_t j = 0; j < 4u; ++j) {
                word |= static_cast<uint32_t>(static_cast<uint8_t>(codes[j])) << (8u * j);
            }
            words_out[b * (kQ8_0BlockSize / 4u) + w] = word;
        }
    }
    return Status::success();
}

[[nodiscard]] Status dequantize_q8_0_matrix(luisa::span<const uint32_t> words,
                                            luisa::span<const float> scales,
                                            uint64_t k, uint64_t n,
                                            luisa::span<float> out) noexcept {
    if (k == 0u || k % kQ8_0BlockSize != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "q8_0 dequantize requires K % 32 == 0", "k");
    }
    uint64_t elements = 0;
    if (!checked_mul(k, n, elements) || static_cast<uint64_t>(out.size()) != elements) {
        return Status::error(StatusCode::invalid_argument,
                             "q8_0 dequant output size mismatch", "out");
    }
    if (static_cast<uint64_t>(words.size()) != elements / 4u) {
        return Status::error(StatusCode::invalid_argument,
                             "q8_0 dequant packed size mismatch", "words");
    }
    const uint64_t groups_per_row = q8_0_group_count(k);
    uint64_t scale_count = 0;
    if (!checked_mul(groups_per_row, n, scale_count) ||
        static_cast<uint64_t>(scales.size()) != scale_count) {
        return Status::error(StatusCode::invalid_argument,
                             "q8_0 dequant scales size mismatch", "scales");
    }
    for (uint64_t row = 0; row < n; ++row) {
        for (uint64_t i = 0; i < k; ++i) {
            const uint64_t global = row * k + i;
            const uint8_t code = static_cast<uint8_t>((words[global / 4u] >> (8u * (global % 4u))) & 0xffu);
            const float scale = scales[row * groups_per_row + i / kQ8_0BlockSize];
            out[global] = int8_to_f32(code) * scale;
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// W4A16 group-wise packed weights (opt_plan/05 Q2) -- host contract.
// ---------------------------------------------------------------------------
namespace {

// Deterministic nearest-int4 quantization of one value against a float scale.
// Rounding is half-away-from-zero on the double quotient, then clamped to the
// [-8, 7] code range. The code is returned as the 4-bit two's-complement
// nibble (i.e. value & 0xF) so packing is a plain OR.
[[nodiscard]] uint32_t w4_quantize_code(double value, double scale) noexcept {
    if (!(scale > 0.0)) { return 0u; }
    const double q = value / scale;
    const double r = q >= 0.0 ? std::floor(q + 0.5) : std::ceil(q - 0.5);
    const double clamped = r > 7.0 ? 7.0 : (r < -8.0 ? -8.0 : r);
    return static_cast<uint32_t>(static_cast<int>(clamped)) & 0xFu;
}

}// namespace

[[nodiscard]] Status validate_w4_geometry(uint64_t k, uint32_t group) noexcept {
    if (group != 32u && group != 64u && group != 128u) {
        return Status::error(StatusCode::invalid_argument,
                             "w4 requires a scale group in {32, 64, 128}", "group");
    }
    if (k == 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "w4 requires K > 0", "k");
    }
    if (k % kW4CodesPerWord != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "w4 requires K % 8 == 0 (whole packed words)", "k");
    }
    if (k % group != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "w4 requires K % group == 0 (whole scale groups)", "k");
    }
    return Status::success();
}

[[nodiscard]] Status W4ClipSearchConfig::validate() const noexcept {
    if (auto s = validate_w4_geometry(group, group); !s.ok()) { return s; }
    if (!(min_ratio > 0.0f) || !(max_ratio <= 1.0f) || min_ratio > max_ratio) {
        return Status::error(StatusCode::invalid_argument,
                             "w4 clip search requires 0 < min_ratio <= max_ratio <= 1",
                             "clip");
    }
    if (steps < 2u) {
        return Status::error(StatusCode::invalid_argument,
                             "w4 clip search requires at least 2 ratio steps", "steps");
    }
    return Status::success();
}

[[nodiscard]] double W4ClipSearchConfig::ratio_at(uint32_t i) const noexcept {
    if (steps <= 1u) { return static_cast<double>(min_ratio); }
    const double lo = static_cast<double>(min_ratio);
    const double hi = static_cast<double>(max_ratio);
    return lo + static_cast<double>(i) * (hi - lo) / static_cast<double>(steps - 1u);
}

namespace {

// The ONE deterministic per-group clipping search of the tree. `importance` is
// either empty (uniform: every channel weighs 1, i.e. the clipping-only search)
// or one measured activation second moment per input channel; the objective and
// its normalization are documented in common.h.
[[nodiscard]] Status pack_w4_matrix_impl(luisa::span<const float> values,
                                         luisa::span<const float> importance,
                                         uint64_t k, uint64_t n,
                                         const W4ClipSearchConfig &cfg,
                                         luisa::span<uint32_t> words_out,
                                         luisa::span<float> scales_out,
                                         W4PackStats *stats) noexcept {
    const bool weighted = !importance.empty();
    uint64_t elements = 0;
    if (!checked_mul(k, n, elements) || static_cast<uint64_t>(values.size()) != elements) {
        return Status::error(StatusCode::invalid_argument,
                             "w4 matrix value count mismatch", "values");
    }
    if (static_cast<uint64_t>(words_out.size()) != elements / kW4CodesPerWord) {
        return Status::error(StatusCode::invalid_argument,
                             "w4 packed output size mismatch (expected elements / 8)", "words_out");
    }
    if (auto s = cfg.validate(); !s.ok()) { return s; }
    const uint64_t gpr = w4_group_count(k, cfg.group);
    uint64_t scale_count = 0;
    if (!checked_mul(gpr, n, scale_count) ||
        static_cast<uint64_t>(scales_out.size()) != scale_count) {
        return Status::error(StatusCode::invalid_argument,
                             "w4 scales output size mismatch", "scales_out");
    }
    if (weighted && static_cast<uint64_t>(importance.size()) != k) {
        return Status::error(StatusCode::invalid_argument,
                             "w4 importance vector must hold exactly K values (one per input "
                             "channel)", "importance");
    }
    const uint32_t group = cfg.group;
    const uint64_t grid = cfg.steps;
    // Fail closed on non-finite input BEFORE writing anything: an int4 code for
    // a NaN/inf value would silently fabricate a weight.
    for (uint64_t i = 0; i < elements; ++i) {
        if (!std::isfinite(values[static_cast<size_t>(i)])) {
            return Status::error(StatusCode::invalid_argument,
                                 "w4 quantization input is not finite", "values");
        }
    }
    if (weighted) {
        for (uint64_t j = 0; j < k; ++j) {
            const float m = importance[static_cast<size_t>(j)];
            if (!std::isfinite(m) || m < 0.0f) {
                return Status::error(StatusCode::invalid_argument,
                                     "w4 importance must be a finite activation second moment "
                                     "(>= 0)", "importance");
            }
        }
    }
    if (k == 0u || n == 0u) { return Status::success(); }
    W4PackStats local{};
    local.groups = gpr * n;
    // Per-group weight sums (the same for every output row). A group whose total
    // weight is zero has no measured activation energy, so its weighted objective
    // is undefined: fail closed rather than invent a weight.
    luisa::vector<double> group_weight(static_cast<size_t>(gpr), 0.0);
    for (uint64_t g = 0; g < gpr; ++g) {
        double sum = 0.0;
        for (uint32_t i = 0; i < group; ++i) {
            sum += weighted ? static_cast<double>(importance[static_cast<size_t>(g * group + i)])
                            : 1.0;
        }
        if (!(sum > 0.0)) {
            return Status::error(StatusCode::invalid_argument,
                                 "w4 importance carries no activation energy for a whole scale "
                                 "group (sum == 0)", "importance");
        }
        group_weight[static_cast<size_t>(g)] = sum;
    }
    double squared_error = 0.0;
    double rtn_squared_error = 0.0;
    double weighted_error = 0.0;
    double rtn_weighted_error = 0.0;
    double ratio_sum = 0.0;
    luisa::vector<float> group_values(static_cast<size_t>(group));
    luisa::vector<uint32_t> best_codes(static_cast<size_t>(group));
    for (uint64_t row = 0; row < n; ++row) {
        for (uint64_t g = 0; g < gpr; ++g) {
            const uint64_t begin = row * k + g * group;
            const double weight_sum = group_weight[static_cast<size_t>(g)];
            double max_abs = 0.0;
            for (uint32_t i = 0; i < group; ++i) {
                const double v = static_cast<double>(values[static_cast<size_t>(begin + i)]);
                group_values[i] = static_cast<float>(v);
                const double a = std::fabs(v);
                if (a > max_abs) { max_abs = a; }
            }
            if (max_abs > local.max_abs) { local.max_abs = max_abs; }
            const double base = max_abs / static_cast<double>(kW4ScaleDenominator);
            uint64_t scale_index = row * gpr + g;
            if (base == 0.0) {
                // All-zero group: scale 0 and all-zero codes (device decode of
                // 0 * 0 is exactly 0, the same as the FP8/Q8_0 zero-group rule).
                scales_out[static_cast<size_t>(scale_index)] = 0.0f;
                for (uint32_t i = 0; i < group; ++i) { best_codes[i] = 0u; }
                for (uint32_t i = 0; i < group; i += kW4CodesPerWord) {
                    uint32_t word = 0u;
                    for (uint32_t j = 0; j < kW4CodesPerWord; ++j) {
                        word |= (best_codes[i + j] & 0xFu) << (4u * j);
                    }
                    words_out[static_cast<size_t>((begin + i) / kW4CodesPerWord)] = word;
                }
                ratio_sum += 1.0;
                continue;
            }
            double best_objective = std::numeric_limits<double>::infinity();
            float best_scale = 0.0f;
            double best_ratio = 1.0;
            for (uint64_t s = 0; s < grid; ++s) {
                const double ratio = cfg.ratio_at(static_cast<uint32_t>(s));
                // The STORED scale is the float one; the search minimizes the
                // error of the value the device will actually decode with.
                const float scale_f = static_cast<float>(base * ratio);
                if (!(scale_f > 0.0f)) { continue; }
                const double scale = static_cast<double>(scale_f);
                double err = 0.0;
                for (uint32_t i = 0; i < group; ++i) {
                    const double v = static_cast<double>(group_values[i]);
                    const double dq = static_cast<double>(w4_code_to_f32(w4_quantize_code(v, scale))) * scale;
                    const double e = v - dq;
                    // Uniform weights reduce to the clipping-only objective.
                    err += (weighted ? static_cast<double>(importance[static_cast<size_t>(g * group + i)])
                                     : 1.0) *
                           e * e;
                }
                const double objective = err / weight_sum;
                if (objective < best_objective) {// strict: ties keep the SMALLER ratio
                    best_objective = objective;
                    best_scale = scale_f;
                    best_ratio = ratio;
                    for (uint32_t i = 0; i < group; ++i) {
                        best_codes[i] = w4_quantize_code(static_cast<double>(group_values[i]),
                                                         static_cast<double>(best_scale));
                    }
                }
            }
            if (!(best_scale > 0.0f)) {
                return Status::error(StatusCode::invalid_argument,
                                     "w4 clip search produced no valid scale", "scale");
            }
            scales_out[static_cast<size_t>(scale_index)] = best_scale;
            // Plain (unweighted) squared error of the CHOSEN scale, so the pack's
            // per-element MSE stays comparable with an unweighted pack, plus the
            // weighted objective that was actually minimized.
            double chosen_sse = 0.0;
            for (uint32_t i = 0; i < group; ++i) {
                const double v = static_cast<double>(group_values[i]);
                const double dq = static_cast<double>(w4_code_to_f32(best_codes[i])) * static_cast<double>(best_scale);
                const double e = v - dq;
                chosen_sse += e * e;
            }
            squared_error += chosen_sse;
            weighted_error += best_objective;
            ratio_sum += best_ratio;
            if (best_ratio < 1.0) { local.clipped_groups++; }
            // Plain RTN reference (the same scale denominator, no clipping) so
            // the clipping gain is measured rather than asserted.
            const double rtn_scale = static_cast<double>(base);
            double rtn_sse = 0.0;
            double rtn_err = 0.0;
            for (uint32_t i = 0; i < group; ++i) {
                const double v = static_cast<double>(group_values[i]);
                const double dq = static_cast<double>(w4_code_to_f32(w4_quantize_code(v, rtn_scale))) * rtn_scale;
                const double e = v - dq;
                rtn_sse += e * e;
                rtn_err += (weighted ? static_cast<double>(importance[static_cast<size_t>(g * group + i)])
                                     : 1.0) *
                           e * e;
            }
            rtn_squared_error += rtn_sse;
            rtn_weighted_error += rtn_err / weight_sum;
            for (uint32_t i = 0; i < group; i += kW4CodesPerWord) {
                uint32_t word = 0u;
                for (uint32_t j = 0; j < kW4CodesPerWord; ++j) {
                    word |= (best_codes[i + j] & 0xFu) << (4u * j);
                }
                words_out[static_cast<size_t>((begin + i) / kW4CodesPerWord)] = word;
            }
        }
    }
    if (stats != nullptr) {
        local.mse = squared_error / static_cast<double>(elements);
        local.rtn_mse = rtn_squared_error / static_cast<double>(elements);
        local.weighted_mse = weighted_error / static_cast<double>(local.groups);
        local.rtn_weighted_mse = rtn_weighted_error / static_cast<double>(local.groups);
        local.mean_ratio = static_cast<float>(ratio_sum / static_cast<double>(local.groups));
        *stats = local;
    }
    return Status::success();
}

}// namespace

[[nodiscard]] Status pack_w4_matrix(luisa::span<const float> values, uint64_t k, uint64_t n,
                                    const W4ClipSearchConfig &cfg,
                                    luisa::span<uint32_t> words_out,
                                    luisa::span<float> scales_out,
                                    W4PackStats *stats) noexcept {
    return pack_w4_matrix_impl(values, {}, k, n, cfg, words_out, scales_out, stats);
}

[[nodiscard]] Status pack_w4_matrix_weighted(luisa::span<const float> values,
                                             luisa::span<const float> importance,
                                             uint64_t k, uint64_t n,
                                             const W4ClipSearchConfig &cfg,
                                             luisa::span<uint32_t> words_out,
                                             luisa::span<float> scales_out,
                                             W4PackStats *stats) noexcept {
    return pack_w4_matrix_impl(values, importance, k, n, cfg, words_out, scales_out, stats);
}

[[nodiscard]] Status dequantize_w4_matrix(luisa::span<const uint32_t> words,
                                          luisa::span<const float> scales,
                                          uint64_t k, uint64_t n, uint32_t group,
                                          luisa::span<float> out) noexcept {
    if (auto s = validate_w4_geometry(k, group); !s.ok()) { return s; }
    uint64_t elements = 0;
    if (!checked_mul(k, n, elements) || static_cast<uint64_t>(out.size()) != elements) {
        return Status::error(StatusCode::invalid_argument,
                             "w4 dequant output size mismatch", "out");
    }
    if (static_cast<uint64_t>(words.size()) != elements / kW4CodesPerWord) {
        return Status::error(StatusCode::invalid_argument,
                             "w4 dequant packed size mismatch", "words");
    }
    const uint64_t gpr = w4_group_count(k, group);
    uint64_t scale_count = 0;
    if (!checked_mul(gpr, n, scale_count) ||
        static_cast<uint64_t>(scales.size()) != scale_count) {
        return Status::error(StatusCode::invalid_argument,
                             "w4 dequant scales size mismatch", "scales");
    }
    for (uint64_t row = 0; row < n; ++row) {
        for (uint64_t i = 0; i < k; ++i) {
            const uint64_t global = row * k + i;
            const uint32_t nibble = (words[static_cast<size_t>(global / kW4CodesPerWord)] >>
                                     (4u * (global % kW4CodesPerWord))) &
                                    0xFu;
            const float scale = scales[static_cast<size_t>(row * gpr + i / group)];
            out[static_cast<size_t>(global)] = w4_code_to_f32(nibble) * scale;
        }
    }
    return Status::success();
}

[[nodiscard]] Result<WeightStorage> classify_w4_pack_storage(
    luisa::string_view name, luisa::span<const uint64_t> dims, uint32_t group) noexcept {
    const bool conv = name.ends_with(".ssm_conv1d.weight");
    if (dims.size() < 2u || conv) {
        // Sensitive tensors stay dense F32 in a W4 pack (contract: norms,
        // ssm_a, ssm_dt.bias and the depthwise convolution are never 4-bit).
        return Result<WeightStorage>::success(WeightStorage::f32_dense);
    }
    if (auto s = validate_w4_geometry(dims[0], group); !s.ok()) {
        return Result<WeightStorage>::failure(
            StatusCode::unsupported,
            luisa::string{"w4 pack geometry: "} + s.message, luisa::string{name});
    }
    return Result<WeightStorage>::success(WeightStorage::w4_packed);
}

}// namespace luisa::qwen
