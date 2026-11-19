// Qwen3.5-4B Tile DSL operators -- implementation.
// See tile_ops.h for the storage ABI, operator inventory and the verified
// Tile/backend constraints. Everything here is host code that either captures
// TileIR (device-free) or computes CPU oracles; no Device/Stream is touched.
//
// Verified device-side idioms (probed on the pinned LuisaCompute, DX + VK):
//  * uint32 decode: division by powers of two and remainder by 65536 work;
//    remainder by 2^24/2^32 and chained div/mul byte extraction miscompile on
//    the DX/VK XIR route, so FP8 byte b is extracted as
//    (words / 2^(8b)) % 65536 (exact: the quotient is < 2^24).
//  * FP8 normal codes are assembled in FP32 integer-valued arithmetic and
//    bitcast once; the disjoint sign bit is added after a float->uint
//    bitcast (integer mul/add on bit patterns miscompiles there).
//  * tile.map bodies are PURE: dynamic gather loads happen through predicated
//    scalar loads inside map (element loads are allowed; tile loads are not).
//  * subtile load/store domains must match the View rank exactly; singleton
//    axes + reshape/relabel adapt ranks (relabel keeps rank, reshape does not).
//  * serial nests carry Scalars AND whole Tiles (GDN state) automatically.
//  * mma needs named-axis contractions: decode W tiles as (n,k), relabel the
//    activation tile to (m,k), contract into (m,n).

#include "qwen/tile_ops.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

#include <luisa/core/mathematics.h>
#include <luisa/tile/algorithms.h>
#include <luisa/tile/verifier.h>

namespace luisa::qwen::tile_ops {

using namespace luisa::compute;
using namespace luisa::compute::tile;

// ---------------------------------------------------------------------------
// Status helpers.
// ---------------------------------------------------------------------------
luisa::string_view tile_op_status_name(TileOpStatus code) noexcept {
    switch (code) {
        case TileOpStatus::ok: return "ok";
        case TileOpStatus::invalid_argument: return "invalid_argument";
        case TileOpStatus::unsupported: return "unsupported";
        case TileOpStatus::capture_failed: return "capture_failed";
        case TileOpStatus::overflow: return "overflow";
    }
    return "unknown";
}

namespace {

[[nodiscard]] TileOpResult invalid(luisa::string message) {
    return TileOpResult::error(TileOpStatus::invalid_argument, std::move(message));
}

[[nodiscard]] TileCapture capture_failure(TileOpStatus code, luisa::string message) {
    TileCapture c;
    c.status = TileOpResult::error(code, std::move(message));
    return c;
}

// Finish a capture: run the verifier, collect diagnostics, fail closed.
[[nodiscard]] TileCapture finish_capture(Kernel kernel) {
    TileCapture c;
    for (auto &d : kernel.diagnostics()) { c.diagnostics.emplace_back(d); }
    if (!kernel.valid()) {
        auto verification = verify(kernel.module());
        for (auto &d : verification.diagnostics()) { c.diagnostics.emplace_back(d.message); }
        c.status = TileOpResult::error(TileOpStatus::capture_failed,
                                       "Tile capture failed verification (see diagnostics)");
        c.kernel = std::move(kernel);
        return c;
    }
    c.kernel = std::move(kernel);
    return c;
}

constexpr uint64_t kMaxExtent = 1ull << 31;// int32-addressable
[[nodiscard]] bool extent_ok(uint64_t v, bool allow_zero = false) noexcept {
    return (allow_zero ? true : v != 0u) && v <= kMaxExtent;
}
[[nodiscard]] bool volume_ok(uint64_t a, uint64_t b) noexcept {
    return a <= kMaxExtent && b <= kMaxExtent / (a == 0 ? 1 : a);
}
[[nodiscard]] bool volume_ok(uint64_t a, uint64_t b, uint64_t c) noexcept {
    return volume_ok(a, b) && c <= kMaxExtent / (a * b == 0 ? 1 : a * b);
}

}// namespace

// ---------------------------------------------------------------------------
// Config / format helpers.
// ---------------------------------------------------------------------------
float QwenTileConfig::attn_scale() const noexcept {
    return 1.0f / std::sqrt(static_cast<float>(attn_head_dim));
}
float QwenTileConfig::gdn_q_scale() const noexcept {
    return 1.0f / std::sqrt(static_cast<float>(gdn_head_dim));
}

luisa::string_view qwen_tile_weight_format_name(QwenTileWeightFormat f) noexcept {
    return f == QwenTileWeightFormat::bf16 ? "bf16" : "fp8_e4m3";
}

TileOpResult validate_qwen_tile_config(const QwenTileConfig &cfg) noexcept {
    if (cfg.hidden == 0 || cfg.ffn == 0 || cfg.vocab == 0) { return invalid("zero model dimension"); }
    if (cfg.q_heads == 0 || cfg.kv_heads == 0 || cfg.q_heads % cfg.kv_heads != 0) {
        return invalid("q_heads must be a positive multiple of kv_heads");
    }
    if (cfg.attn_head_dim == 0 || cfg.attn_head_dim % 2 != 0) { return invalid("attn_head_dim must be even"); }
    if (cfg.rope_dim == 0 || cfg.rope_dim % 2 != 0 || cfg.rope_dim > cfg.attn_head_dim) {
        return invalid("rope_dim must be even and <= attn_head_dim");
    }
    if (!(cfg.rope_theta > 0.0f) || !std::isfinite(cfg.rope_theta)) { return invalid("rope_theta must be finite > 0"); }
    if (cfg.gdn_qk_heads == 0 || cfg.gdn_v_heads == 0 || cfg.gdn_v_heads % cfg.gdn_qk_heads != 0) {
        return invalid("gdn_v_heads must be a positive multiple of gdn_qk_heads");
    }
    if (cfg.gdn_head_dim == 0) { return invalid("zero gdn_head_dim"); }
    if (cfg.conv_kernel < 2) { return invalid("conv_kernel must be >= 2 (ring history)"); }
    if (!(cfg.norm_eps > 0.0f) || !(cfg.l2_eps > 0.0f)) { return invalid("eps must be > 0"); }
    if (!(cfg.softplus_threshold > 0.0f)) { return invalid("softplus_threshold must be > 0"); }
    return TileOpResult::success();
}

uint64_t tile_packed_matrix_words(QwenTileWeightFormat f, uint64_t k, uint64_t n) noexcept {
    if (k > kMaxExtent || n > kMaxExtent / (k == 0 ? 1 : k)) { return 0; }
    auto elements = k * n;
    if (f == QwenTileWeightFormat::bf16) {
        if (elements % 2 != 0) { return 0; }// contract: even K for this model
        return elements / 2;
    }
    return (elements + 3) / 4;
}
uint64_t tile_fp8_scale_elements(uint64_t k, uint64_t n) noexcept {
    auto groups = (k + kTileFp8Group - 1) / kTileFp8Group;
    if (groups > kMaxExtent || n > kMaxExtent / (groups == 0 ? 1 : groups)) { return 0; }
    return n * groups;
}

// ---------------------------------------------------------------------------
// Host codecs (bit-exact; independent of the device decode arithmetic).
// ---------------------------------------------------------------------------
uint16_t tile_host_f32_to_bf16_bits(float v) noexcept {
    auto bits = std::bit_cast<uint32_t>(v);
    if (((bits >> 23) & 0xffu) == 0xffu && (bits & 0x7fffffu) != 0u) {
        return static_cast<uint16_t>((bits >> 16) | 0x0040u);// quiet BF16 NaN
    }
    auto rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(rounded >> 16);
}
float tile_host_bf16_bits_to_f32(uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}
float tile_host_fp8_e4m3_bits_to_f32(uint8_t code) noexcept {
    auto sign = static_cast<uint32_t>(code & 0x80u) << 24;
    auto mag = static_cast<uint32_t>(code & 0x7fu);
    auto e = mag >> 3, m = mag & 7u;
    if (mag == 0x7fu) { return std::bit_cast<float>(sign | 0x7fc00000u); }
    if (e == 0u) {
        auto v = static_cast<float>(m) * 0x1p-9f;
        return std::bit_cast<float>(std::bit_cast<uint32_t>(v) | sign);
    }
    return std::bit_cast<float>(sign | ((e + 120u) << 23) | (m << 20));
}
uint8_t tile_host_f32_to_fp8_e4m3_bits(float v) noexcept {
    auto bits = std::bit_cast<uint32_t>(v);
    auto sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    auto mag = bits & 0x7fffffffu;
    if (mag > 0x7f800000u) { return static_cast<uint8_t>(sign | 0x7fu); }// NaN
    if (mag >= 0x43e80000u) { return static_cast<uint8_t>(sign | 0x7eu); }// >=464 saturates to 448
    if (mag < 0x3c800000u) {
        // Subnormal range: |v| < 2^-6 (fp8 e8=1 starts at 2^-6; codes
        // 0x01..0x07 and the carry 0x08 all live below it). Round |v| / 2^-9
        // to nearest even int in DOUBLE: the float product v*512 can itself
        // round (e.g. 7*2^-9 -> exactly 4.0f) and lose the tie direction.
        // m == 8 carries into the smallest normal 0x08.
        auto scaled = static_cast<double>(std::bit_cast<float>(mag)) * 512.0;
        auto floor_v = std::floor(scaled);
        auto frac = scaled - floor_v;
        auto m = static_cast<uint32_t>(floor_v);
        if (frac > 0.5 || (frac == 0.5 && (m & 1u) != 0u)) { m += 1; }
        return static_cast<uint8_t>(sign | static_cast<uint8_t>(m));
    }
    auto e32 = (mag >> 23) & 0xffu;
    auto mant = mag & 0x7fffffu;
    auto e8 = static_cast<int32_t>(e32) - 120;
    auto keep = mant >> 20;
    auto rem = mant & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (keep & 1u) != 0u)) {
        keep += 1;
        if (keep == 8u) {
            keep = 0;
            e8 += 1;
        }
    }
    if (e8 > 15 || (e8 == 15 && keep >= 7)) { return static_cast<uint8_t>(sign | 0x7eu); }
    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(e8) << 3) | keep);
}

TileOpResult tile_host_pack_matrix(QwenTileWeightFormat format, luisa::span<const float> w,
                                   uint64_t k, uint64_t n,
                                   luisa::vector<uint32_t> &words, luisa::vector<float> &scales) {
    if (k == 0 || n == 0) { return invalid("zero matrix dimension"); }
    if (!volume_ok(k, n) || w.size() < k * n) { return invalid("matrix span too small / overflow"); }
    auto word_count = tile_packed_matrix_words(format, k, n);
    if (word_count == 0) { return invalid("packed word count overflow (bf16 needs even k*n)"); }
    words.assign(word_count, 0u);
    if (format == QwenTileWeightFormat::bf16) {
        scales.clear();
        for (uint64_t i = 0; i < k * n; i++) {
            words[i / 2] |= static_cast<uint32_t>(tile_host_f32_to_bf16_bits(w[i])) << ((i % 2) * 16);
        }
        return TileOpResult::success();
    }
    auto scale_count = tile_fp8_scale_elements(k, n);
    if (scale_count == 0) { return invalid("scale count overflow"); }
    scales.assign(scale_count, 0.0f);
    auto groups = (k + kTileFp8Group - 1) / kTileFp8Group;
    for (uint64_t row = 0; row < n; row++) {
        for (uint64_t g = 0; g < groups; g++) {
            auto begin = g * kTileFp8Group;
            auto end = std::min<uint64_t>(k, begin + kTileFp8Group);
            float max_abs = 0.0f;
            for (auto kk = begin; kk < end; kk++) {
                auto v = w[row * k + kk];
                if (!std::isfinite(v)) { return invalid("non-finite weight in fp8 conversion"); }
                max_abs = std::fmax(max_abs, std::fabs(v));
            }
            auto scale = max_abs / 448.0f;
            scales[row * groups + g] = scale;
            for (auto kk = begin; kk < end; kk++) {
                auto v = w[row * k + kk];
                auto code = scale > 0.0f ? tile_host_f32_to_fp8_e4m3_bits(v / scale) : uint8_t{0};
                auto index = row * k + kk;
                words[index / 4] |= static_cast<uint32_t>(code) << ((index % 4) * 8);
            }
        }
    }
    return TileOpResult::success();
}

float tile_host_decode_packed(QwenTileWeightFormat format, luisa::span<const uint32_t> words,
                              luisa::span<const float> scales, uint64_t k, uint64_t index) noexcept {
    if (format == QwenTileWeightFormat::bf16) {
        auto word = words[index / 2];
        auto code = static_cast<uint16_t>((word >> ((index % 2) * 16)) & 0xffffu);
        return tile_host_bf16_bits_to_f32(code);
    }
    auto word = words[index / 4];
    auto code = static_cast<uint8_t>((word >> ((index % 4) * 8)) & 0xffu);
    auto row = index / k;
    auto group = (index % k) / kTileFp8Group;
    return tile_host_fp8_e4m3_bits_to_f32(code) * scales[row * ((k + kTileFp8Group - 1) / kTileFp8Group) + group];
}

// ---------------------------------------------------------------------------
// Shared device-side building blocks (capture-time helpers; pure Tile code).
// ---------------------------------------------------------------------------
namespace {

[[nodiscard]] Tile<float> tile_silu(const Tile<float> &x) {
    return x / (1.0f + exp(-x));
}
[[nodiscard]] Scalar<float> scalar_silu(const Scalar<float> &x) {
    return x / (1.0f + exp(-x));
}
[[nodiscard]] Scalar<float> scalar_sigmoid(const Scalar<float> &x) {
    return 1.0f / (1.0f + exp(-x));
}
[[nodiscard]] Tile<float> tile_sigmoid(const Tile<float> &x) {
    return 1.0f / (1.0f + exp(-x));
}
[[nodiscard]] Scalar<float> scalar_softplus(const Scalar<float> &x, float threshold) {
    auto base = log(1.0f + exp(-abs(x))) + max(x, 0.0f);
    return ite(x > threshold, x, base);
}

// Cody-Waite reduction to [-pi/4, pi/4] plus bounded Taylor series. TileIR
// has no sin/cos opcode; this deterministic portable evaluation is shared by
// the kernels and the CPU oracle (tile_host_rope_sin_cos).
[[nodiscard]] Scalar<float> scalar_sin_reduced(const Scalar<float> &r) {
    auto r2 = r * r;
    return r * (1.0f + r2 * (-1.0f / 6.0f + r2 * (1.0f / 120.0f + r2 * (-1.0f / 5040.0f + r2 * (1.0f / 362880.0f)))));
}
[[nodiscard]] Scalar<float> scalar_cos_reduced(const Scalar<float> &r) {
    auto r2 = r * r;
    return 1.0f + r2 * (-0.5f + r2 * (1.0f / 24.0f + r2 * (-1.0f / 720.0f + r2 * (1.0f / 40320.0f))));
}

struct SinCos {
    Scalar<float> sin_v;
    Scalar<float> cos_v;
};
[[nodiscard]] SinCos scalar_sin_cos(const Scalar<float> &x) {
    constexpr float two_over_pi = 0.6366197723675814f;
    constexpr float pio2_hi = 1.5707962512969971f;
    constexpr float pio2_mid = 7.5497894158615964e-08f;
    constexpr float pio2_lo = 5.3903025299577184e-15f;
    auto qf = x * two_over_pi;
    auto q = cast<int32_t>(qf + ite(qf >= 0.0f, 0.5f, -0.5f));
    auto qfl = cast<float>(q);
    auto r = ((x - qfl * pio2_hi) - qfl * pio2_mid) - qfl * pio2_lo;
    auto quadrant = q % 4;
    quadrant = ite(quadrant < 0, quadrant + 4, quadrant);
    auto s = scalar_sin_reduced(r);
    auto c = scalar_cos_reduced(r);
    auto sin_v = ite(quadrant == 0, s, ite(quadrant == 1, c, ite(quadrant == 2, -s, -c)));
    auto cos_v = ite(quadrant == 0, c, ite(quadrant == 1, -s, ite(quadrant == 2, -c, s)));
    return {sin_v, cos_v};
}

// BF16 pair decode: one uint32 word -> (lo, hi) FP32 values.
struct Bf16Pair {
    Tile<float> lo;
    Tile<float> hi;
};
[[nodiscard]] Bf16Pair decode_bf16_words(const Tile<uint32_t> &words) {
    auto low_bits = words % 65536u;
    auto lo = bitcast<float>(low_bits * 65536u);
    auto hi = bitcast<float>(words - low_bits);
    return {lo, hi};
}
[[nodiscard]] Scalar<float> decode_bf16_word_scalar(const Scalar<uint32_t> &word, bool high) {
    auto low_bits = word % 65536u;
    auto bits = high ? word - low_bits : low_bits * 65536u;
    return bitcast<float>(bits);
}

// FP8 E4M3FN decode of one byte lane (0..3) of each uint32 word.
// Byte extraction: (words / 2^(8*lane)) % 65536 -- verified exact on DX/VK.
[[nodiscard]] Tile<float> decode_fp8_lane(const Tile<uint32_t> &words, uint32_t lane) {
    auto shift = 1u << (8 * lane);
    auto code = (words / shift) % 65536u;
    // Sign bit via INTEGER arithmetic: cast<uint32>(2^31 as float) overflows
    // the signed conversion on the XIR route and yields 0, dropping the sign.
    auto sign_bits = (code / 128u) * 2147483648u;
    auto mag = code % 128u;
    auto e = mag / 8u;
    auto m = mag % 8u;
    auto e_f = cast<float>(e);
    auto m_f = cast<float>(m);
    auto normal_mag = e_f * 8388608.0f + (1006632960.0f + m_f * 1048576.0f);
    auto normal = bitcast<float>(cast<uint32_t>(normal_mag) + sign_bits);
    // Subnormal: the FP32 BITS of the value m*2^-9 (bitcast, NOT a truncating
    // cast<uint32> which would zero every value < 1).
    auto sub = bitcast<float>(bitcast<uint32_t>(m_f * 0x1p-9f) + sign_bits);
    auto value = ite(e == 0u, sub, normal);
    return ite(mag == 127u, bitcast<float>(sign_bits + 2143289344u), value);
}

// Interleave four decoded FP8 byte lanes into one value per channel.
[[nodiscard]] Tile<float> fp8_quad_values(const Tile<uint32_t> &words, Axis channel, Axis word_axis) {
    auto b0 = decode_fp8_lane(words, 0);
    auto b1 = decode_fp8_lane(words, 1);
    auto b2 = decode_fp8_lane(words, 2);
    auto b3 = decode_fp8_lane(words, 3);
    return map<float>(shape(channel), [&](const Nest &e) {
        auto j = e.index(channel);
        auto w = j / 4;
        auto lane = j % 4;
        return ite(lane == 0, b0.at(coord(w)),
                   ite(lane == 1, b1.at(coord(w)),
                       ite(lane == 2, b2.at(coord(w)), b3.at(coord(w)))));
    });
}
[[nodiscard]] Tile<float> bf16_pair_values(const Tile<uint32_t> &words, Axis channel) {
    auto pair = decode_bf16_words(words);
    return map<float>(shape(channel), [&](const Nest &e) {
        auto j = e.index(channel);
        return ite(j % 2 == 0, pair.lo.at(coord(j / 2)), pair.hi.at(coord(j / 2)));
    });
}


// Assemble a rank-1 Tile from per-element scalar values computed in a serial
// nest (tile.map is pure: loads must stay in the enclosing execution nest).
// The accumulator starts zeroed; every slot is written exactly once.
template<typename F>
[[nodiscard]] Tile<float> serial_row(Axis element_axis, F &&body) {
    auto values = zeros<float>(shape(element_axis));
    for (auto &e : serial(shape(element_axis))) {
        auto v = body(e.index());
        values = map<float>(shape(element_axis), [&](const Nest &m) {
            return ite(m.index(element_axis) == e.index(), v, values.at(coord(m.index(element_axis))));
        });
    }
    return values;
}

}// namespace

void tile_host_rope_sin_cos(float angle, float &sin_out, float &cos_out) noexcept {
    auto f32 = [](double v) { return static_cast<float>(v); };
    constexpr float two_over_pi = 0.6366197723675814f;
    constexpr float pio2_hi = 1.5707962512969971f;
    constexpr float pio2_mid = 7.5497894158615964e-08f;
    constexpr float pio2_lo = 5.3903025299577184e-15f;
    auto qf = f32(static_cast<double>(angle) * two_over_pi);
    auto q = static_cast<int32_t>(qf + (qf >= 0.0f ? 0.5f : -0.5f));
    auto qfl = static_cast<float>(q);
    auto r = f32(f32(f32(angle - f32(qfl * pio2_hi)) - f32(qfl * pio2_mid)) - f32(qfl * pio2_lo));
    auto quadrant = q % 4;
    if (quadrant < 0) { quadrant += 4; }
    double r2 = static_cast<double>(r) * r;
    auto s = f32(r * f32(1.0f + f32(r2 * f32(-1.0f / 6.0f + f32(r2 * f32(1.0f / 120.0f + f32(r2 * f32(-1.0f / 5040.0f + f32(r2 * (1.0f / 362880.0f))))))))));
    auto c = f32(1.0f + f32(r2 * f32(-0.5f + f32(r2 * f32(1.0f / 24.0f + f32(r2 * f32(-1.0f / 720.0f + f32(r2 * (1.0f / 40320.0f)))))))));
    switch (quadrant) {
        case 0: sin_out = s; cos_out = c; break;
        case 1: sin_out = c; cos_out = f32(-s); break;
        case 2: sin_out = f32(-s); cos_out = f32(-c); break;
        default: sin_out = f32(-c); cos_out = s; break;
    }
}

// ---------------------------------------------------------------------------
// 1. Embedding gather.
// ---------------------------------------------------------------------------
TileCapture capture_tile_embedding(QwenTileWeightFormat format, uint32_t tokens,
                                   uint32_t hidden, uint32_t vocab) {
    if (!extent_ok(tokens) || !extent_ok(hidden) || !extent_ok(vocab)) {
        return capture_failure(TileOpStatus::invalid_argument, "embedding extents out of range");
    }
    if (format == QwenTileWeightFormat::bf16 && hidden % 2 != 0) {
        return capture_failure(TileOpStatus::invalid_argument, "bf16 embedding requires even hidden");
    }
    if (format == QwenTileWeightFormat::fp8_e4m3 && (hidden % 4 != 0 || hidden % kTileFp8Group != 0)) {
        return capture_failure(TileOpStatus::invalid_argument,
                               "fp8 embedding requires hidden divisible by 64 (whole words and scale groups per row)");
    }
    if (!volume_ok(vocab, hidden)) {
        return capture_failure(TileOpStatus::overflow, "embedding table volume overflow");
    }
    const auto T = static_cast<int64_t>(tokens);
    const auto H = static_cast<int64_t>(hidden);
    const auto V = static_cast<int64_t>(vocab);
    const auto groups = (H + kTileFp8Group - 1) / kTileFp8Group;
    constexpr int64_t CB = 128;
    auto definition = tile_kernel(
        format == QwenTileWeightFormat::bf16 ? "qwen_tile_embedding_bf16" : "qwen_tile_embedding_fp8",
        [=](TensorView<const int32_t, 1> ids,
            TensorView<const uint32_t, 1> table,
            TensorView<const float, 1> scales,
            TensorView<float, 2> out) {
            auto c = axis("c", CB);
            auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
            for (auto &tok : parallel(shape(T))) {
                auto id = ids(tok.index()).load();
                auto valid = (id >= 0) && (id < static_cast<int32_t>(V));
                auto row = cast<int64_t>(ite(valid, id, 0));
                for (auto &cb : serial(shape(ceil_div(H, CB)))) {
                    auto c0 = cb.index() * CB;
                    if (format == QwenTileWeightFormat::bf16) {
                        auto c2 = axis("c2", CB / 2);
                        // The selected row and channel block are contiguous in
                        // the packed ABI. Load them as one tile instead of the
                        // old O(CB^2) serial select-scan, which also created a
                        // large per-worker XIR snapshot on DX/VK.
                        auto words = table.tile(coord(row * (H / 2) + c0 / 2), shape(c2)).load();
                        auto values = bf16_pair_values(words, c);
                        values = map<float>(shape(c), [&](const Nest &e) {
                            return ite(valid && (c0 + e.index(c) < H), values.at(coord(e.index(c))), 0.0f);
                        });
                        out(coord(tok.index(), c0), shape(one, c)).store(reshape(values, shape(one, c)));
                    } else {
                        auto c4 = axis("c4", CB / 4);
                        auto sg = axis("sg", CB / kTileFp8Group);
                        auto words = table.tile(coord(row * (H / 4) + c0 / 4), shape(c4)).load();
                        auto raw = fp8_quad_values(words, c, c4);
                        auto sv = scales.tile(coord(row * groups + c0 / kTileFp8Group), shape(sg)).load();
                        auto values = map<float>(shape(c), [&](const Nest &e) {
                            auto cc = c0 + e.index(c);
                            auto s = sv.at(coord(e.index(c) / kTileFp8Group));
                            return ite(valid && (cc < H), raw.at(coord(e.index(c))) * s, 0.0f);
                        });
                        out(coord(tok.index(), c0), shape(one, c)).store(reshape(values, shape(one, c)));
                    }
                }
            }
        });
    if (format == QwenTileWeightFormat::bf16) {
        return finish_capture(definition.capture(
            tensor_shape(T), tensor_shape(V * (H / 2)), tensor_shape(1), tensor_shape(T, H)));
    }
    return finish_capture(definition.capture(
        tensor_shape(T), tensor_shape(V * (H / 4)), tensor_shape(V * groups), tensor_shape(T, H)));
}

// ---------------------------------------------------------------------------
// 2. Zero-centered RMSNorm.
// ---------------------------------------------------------------------------
TileCapture capture_tile_rmsnorm(TileNormEpilogue epilogue, uint32_t rows, uint32_t dim, float eps) {
    if (!extent_ok(rows) || !extent_ok(dim) || !volume_ok(rows, dim)) {
        return capture_failure(TileOpStatus::invalid_argument, "rmsnorm extents out of range");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        return capture_failure(TileOpStatus::invalid_argument, "rmsnorm eps must be finite > 0");
    }
    const auto R = static_cast<int64_t>(rows);
    const auto D = static_cast<int64_t>(dim);
    const auto inv_dim = 1.0f / static_cast<float>(dim);
    const auto has_residual = epilogue == TileNormEpilogue::residual;
    auto definition = tile_kernel(
        has_residual ? "qwen_tile_rmsnorm_residual" : "qwen_tile_rmsnorm",
        [=](TensorView<const float, 2> x,
            TensorView<const float, 1> w,
            TensorView<const float, 2> residual,
            TensorView<float, 2> y,
            TensorView<float, 2> sum_out) {
            auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
            auto d = axis("d", D);
            for (auto &row : parallel(shape(R))) {
                auto xv = x.tile(coord(row.index(), 0), shape(one, d)).load();
                auto s = xv;
                if (has_residual) {
                    s = xv + residual.tile(coord(row.index(), 0), shape(one, d)).load();
                    sum_out(coord(row.index(), 0), shape(one, d)).store(s);
                }
                auto mean_square = reduce(s * s, d, add, reduction::fold_left) * inv_dim;
                auto scale = 1.0f / sqrt(mean_square + eps);
                auto wv = reshape(w.tile(coord(0), shape(d)).load(), shape(one, d));
                auto normed = s * scale * wv;
                y(coord(row.index(), 0), shape(one, d)).store(normed);
            }
        });
    auto dummy = tensor_shape(1, 1);
    if (has_residual) {
        return finish_capture(definition.capture(
            tensor_shape(R, D), tensor_shape(D), tensor_shape(R, D), tensor_shape(R, D), tensor_shape(R, D)));
    }
    return finish_capture(definition.capture(
        tensor_shape(R, D), tensor_shape(D), dummy, tensor_shape(R, D), dummy));
}

// ---------------------------------------------------------------------------
// 3. Packed GEMM/GEMV.
// ---------------------------------------------------------------------------
namespace {

// Decode-time matrix-vector product. Unlike the general GEMM this does not pad
// M=1 to a 16-row MMA tile. One program owns TN output columns and streams K in
// 64-element blocks through the ordinary tensor-core contraction path. The
// logical one-row activation and small accumulator remove the large serial-MMA
// snapshots that made the DX/VK XIR route fragile without introducing a slow
// scalar reduction realization.
[[nodiscard]] TileCapture capture_tile_gemv_impl(
    QwenTileWeightFormat format, TileMatmulEpilogue epilogue,
    const TileGemmShape &gemm_shape, uint32_t k_offset, uint32_t k_seg,
    bool partial) {
    const auto N = static_cast<int64_t>(gemm_shape.n);
    const auto K = static_cast<int64_t>(gemm_shape.k);
    const auto TN = static_cast<int64_t>(gemm_shape.tile_n);
    const auto KO = static_cast<int64_t>(k_offset);
    const auto KS = static_cast<int64_t>(k_seg);
    constexpr int64_t BK = kTileGemmKBlock;
    if (gemm_shape.m != 1u || N <= 0 || K <= 0 || TN <= 0 || KS <= 0 ||
        !extent_ok(gemm_shape.n) || !extent_ok(gemm_shape.k)) {
        return capture_failure(TileOpStatus::invalid_argument, "gemv extents out of range");
    }
    if (TN > 256 || (partial && (KO % BK != 0 || KS % BK != 0)) || KO < 0 || KO + KS > K) {
        return capture_failure(TileOpStatus::invalid_argument,
                               "gemv requires a valid tile and block-aligned in-range K segment");
    }
    if (format == QwenTileWeightFormat::bf16 && K % 2 != 0) {
        return capture_failure(TileOpStatus::invalid_argument, "bf16 gemv requires even K");
    }
    if (format == QwenTileWeightFormat::fp8_e4m3 && K % 4 != 0) {
        return capture_failure(TileOpStatus::invalid_argument, "fp8 gemv requires K divisible by 4");
    }
    if (!volume_ok(gemm_shape.n, gemm_shape.k)) {
        return capture_failure(TileOpStatus::overflow, "gemv volume overflow");
    }
    const auto groups = (K + BK - 1) / BK;
    const auto k_blocks = (KS + BK - 1) / BK;
    const auto ko_blocks = KO / BK;
    const auto has_residual = !partial && epilogue == TileMatmulEpilogue::residual;
    auto definition = tile_kernel(
        format == QwenTileWeightFormat::bf16 ?
            (partial ? "qwen_tile_gemv_bf16_kseg" : "qwen_tile_gemv_bf16") :
            (partial ? "qwen_tile_gemv_fp8_kseg" : "qwen_tile_gemv_fp8"),
        [=](TensorView<const float, 2> x,
            TensorView<const uint32_t, 2> w,
            TensorView<const float, 2> scales,
            TensorView<const float, 2> residual,
            TensorView<float, 2> y) {
            auto one = axis("one", 1);
            auto k = axis("k", BK);
            auto n = axis("n", TN);
            auto gn = axis("gn", ceil_div(N, TN));
            for (auto &g : parallel(shape(gn))) {
                auto n0 = g.index(gn) * TN;
                auto acc = zeros<float>(shape(one, n));
                for (auto &kb : serial(shape(k_blocks))) {
                    auto k0 = KO + kb.index() * BK;
                    auto a = x.tile(coord(0, k0), shape(one, k)).load();
                    if (format == QwenTileWeightFormat::bf16) {
                        auto k2 = axis("k2", BK / 2);
                        auto words = w.tile(coord(n0, k0 / 2), shape(n, k2)).load();
                        auto pair = decode_bf16_words(words);
                        auto a_lo = map<float>(shape(one, k2), [&](const Nest &e) {
                            return a.at(coord(0, e.index(k2) * 2));
                        });
                        auto a_hi = map<float>(shape(one, k2), [&](const Nest &e) {
                            return a.at(coord(0, e.index(k2) * 2 + 1));
                        });
                        if (gemm_shape.gemv_mma) {
                            acc = mma(a_lo, pair.lo, acc, MmaPolicy{.allow_reassociation = false});
                            acc = mma(a_hi, pair.hi, acc, MmaPolicy{.allow_reassociation = false});
                        } else {
                            auto products = map<float>(shape(n, k2), [&](const Nest &e) {
                                auto nn = e.index(n);
                                auto ww = e.index(k2);
                                return a_lo.at(coord(0, ww)) * pair.lo.at(coord(nn, ww)) +
                                       a_hi.at(coord(0, ww)) * pair.hi.at(coord(nn, ww));
                            });
                            acc = acc + reshape(reduce(products, k2, add), shape(one, n));
                        }
                    } else {
                        auto k4 = axis("k4", BK / 4);
                        auto words = w.tile(coord(n0, k0 / 4), shape(n, k4)).load();
                        auto b0 = decode_fp8_lane(words, 0);
                        auto b1 = decode_fp8_lane(words, 1);
                        auto b2 = decode_fp8_lane(words, 2);
                        auto b3 = decode_fp8_lane(words, 3);
                        auto sc = scales.tile(coord(n0, ko_blocks + kb.index()),
                                              shape(n, axis("s1", 1))).load();
                        auto decoded = map<float>(shape(n, k), [&](const Nest &e) {
                            auto nn = e.index(n);
                            auto kk = e.index(k);
                            auto w4 = kk / 4;
                            auto lane = kk % 4;
                            auto v = ite(lane == 0, b0.at(coord(nn, w4)),
                                         ite(lane == 1, b1.at(coord(nn, w4)),
                                             ite(lane == 2, b2.at(coord(nn, w4)), b3.at(coord(nn, w4)))));
                            return v * sc.at(coord(nn, 0));
                        });
                        if (gemm_shape.gemv_mma) {
                            acc = mma(a, decoded, acc, MmaPolicy{.allow_reassociation = false});
                        } else {
                            auto products = map<float>(shape(n, k), [&](const Nest &e) {
                                return a.at(coord(0, e.index(k))) *
                                       decoded.at(coord(e.index(n), e.index(k)));
                            });
                            acc = acc + reshape(reduce(products, k, add), shape(one, n));
                        }
                    }
                }
                if (has_residual) {
                    acc = acc + residual.tile(coord(0, n0), shape(one, n)).load();
                }
                y(coord(0, n0), shape(one, n)).store(acc);
            }
        });
    auto dummy = tensor_shape(1, 1);
    auto w_shape = format == QwenTileWeightFormat::bf16 ?
                       tensor_shape(N, K / 2) : tensor_shape(N, K / 4);
    auto s_shape = format == QwenTileWeightFormat::bf16 ? dummy : tensor_shape(N, groups);
    auto r_shape = has_residual ? tensor_shape(1, N) : dummy;
    return finish_capture(definition.capture(
        tensor_shape(1, K), w_shape, s_shape, r_shape, tensor_shape(1, N)));
}

}// namespace

TileCapture capture_tile_gemm(QwenTileWeightFormat format, TileMatmulEpilogue epilogue,
                              const TileGemmShape &gemm_shape) {
    if (gemm_shape.m == 1u) {
        return capture_tile_gemv_impl(format, epilogue, gemm_shape, 0u, gemm_shape.k, false);
    }
    const auto M = static_cast<int64_t>(gemm_shape.m);
    const auto N = static_cast<int64_t>(gemm_shape.n);
    const auto K = static_cast<int64_t>(gemm_shape.k);
    const auto TM = static_cast<int64_t>(gemm_shape.tile_m);
    const auto TN = static_cast<int64_t>(gemm_shape.tile_n);

    if (M <= 0 || N <= 0 || K <= 0 || TM <= 0 || TN <= 0 ||
        !extent_ok(gemm_shape.m) || !extent_ok(gemm_shape.n) || !extent_ok(gemm_shape.k)) {
        return capture_failure(TileOpStatus::invalid_argument, "gemm extents out of range");
    }
    if (TM > 256 || TN > 256 || TM * TN > 16384) {
        return capture_failure(TileOpStatus::invalid_argument, "gemm tile too large (<=16K elements)");
    }
    if (format == QwenTileWeightFormat::bf16 && K % 2 != 0) {
        return capture_failure(TileOpStatus::invalid_argument, "bf16 gemm requires even K");
    }
    // The packed FP8 ABI is element-contiguous (4 codes/uint32, low-first).
    // The Tile GEMM loads weights as a 2D (n, K/4) word tile, which assumes
    // each output row starts on a word boundary. That holds iff K % 4 == 0.
    if (format == QwenTileWeightFormat::fp8_e4m3 && K % 4 != 0) {
        return capture_failure(TileOpStatus::invalid_argument,
                               "fp8 gemm requires K divisible by 4 (word-aligned rows)");
    }
    if (!volume_ok(gemm_shape.m, gemm_shape.k) || !volume_ok(gemm_shape.n, gemm_shape.k) || !volume_ok(gemm_shape.m, gemm_shape.n)) {
        return capture_failure(TileOpStatus::overflow, "gemm volume overflow");
    }
    constexpr int64_t BK = kTileGemmKBlock;// one FP8 scale group per step
    const auto groups = (K + BK - 1) / BK;
    const auto has_residual = epilogue == TileMatmulEpilogue::residual;
    const auto k_blocks = ceil_div(K, BK);
    auto definition = tile_kernel(
        format == QwenTileWeightFormat::bf16 ? "qwen_tile_gemm_bf16" : "qwen_tile_gemm_fp8",
        [=](TensorView<const float, 2> x,
            TensorView<const uint32_t, 2> w,
            TensorView<const float, 2> scales,
            TensorView<const float, 2> residual,
            TensorView<float, 2> y) {
            auto m = axis("m", TM);
            auto n = axis("n", TN);
            auto k = axis("k", BK);
            auto gm = axis("gm", ceil_div(M, TM));
            auto gn = axis("gn", ceil_div(N, TN));
            for (auto &g : parallel(shape(gm, gn))) {
                auto m0 = g.index(gm) * TM;
                auto n0 = g.index(gn) * TN;
                auto acc = zeros<float>(shape(m, n));
                for (auto &kb : serial(shape(k_blocks))) {
                    auto k0 = kb.index() * BK;
                    auto a = x.tile(coord(m0, k0), shape(m, k)).load();
                    if (format == QwenTileWeightFormat::bf16) {
                        auto k2 = axis("k2", BK / 2);
                        auto words = w.tile(coord(n0, k0 / 2), shape(n, k2)).load();
                        auto pair = decode_bf16_words(words);
                        auto a_lo = map<float>(shape(m, k2), [&](const Nest &e) {
                            return a.at(coord(e.index(m), e.index(k2) * 2));
                        });
                        auto a_hi = map<float>(shape(m, k2), [&](const Nest &e) {
                            return a.at(coord(e.index(m), e.index(k2) * 2 + 1));
                        });
                        acc = mma(a_lo, pair.lo, acc, MmaPolicy{.allow_reassociation = false});
                        acc = mma(a_hi, pair.hi, acc, MmaPolicy{.allow_reassociation = false});
                    } else {
                        auto k4 = axis("k4", BK / 4);
                        auto words = w.tile(coord(n0, k0 / 4), shape(n, k4)).load();
                        auto sc = reshape(scales.tile(coord(n0, kb.index()), shape(n, axis("s1", 1))).load(), shape(n));
                        auto b0 = decode_fp8_lane(words, 0);
                        auto b1 = decode_fp8_lane(words, 1);
                        auto b2 = decode_fp8_lane(words, 2);
                        auto b3 = decode_fp8_lane(words, 3);
                        auto decoded = map<float>(shape(n, k), [&](const Nest &e) {
                            auto nn = e.index(n);
                            auto kk = e.index(k);
                            auto w4 = kk / 4;
                            auto lane = kk % 4;
                            auto v = ite(lane == 0, b0.at(coord(nn, w4)),
                                         ite(lane == 1, b1.at(coord(nn, w4)),
                                             ite(lane == 2, b2.at(coord(nn, w4)), b3.at(coord(nn, w4)))));
                            return v * sc.at(coord(nn));
                        });
                        acc = mma(a, decoded, acc, MmaPolicy{.allow_reassociation = false});
                    }
                }
                if (has_residual) {
                    acc = acc + residual.tile(coord(m0, n0), shape(m, n)).load();
                }
                y(coord(m0, n0), shape(m, n)).store(acc);
            }
        });
    auto dummy = tensor_shape(1, 1);
    auto w_shape = format == QwenTileWeightFormat::bf16 ?
                       tensor_shape(N, K / 2) : tensor_shape(N, K / 4);
    auto s_shape = format == QwenTileWeightFormat::bf16 ? dummy : tensor_shape(N, groups);
    auto r_shape = has_residual ? tensor_shape(M, N) : dummy;
    return finish_capture(definition.capture(
        tensor_shape(M, K), w_shape, s_shape, r_shape, tensor_shape(M, N)));
}

TileCapture capture_tile_gemm_kseg(QwenTileWeightFormat format, const TileGemmShape &gemm_shape,
                                   uint32_t k_offset, uint32_t k_seg) {
    if (gemm_shape.m == 1u) {
        return capture_tile_gemv_impl(format, TileMatmulEpilogue::none, gemm_shape,
                                      k_offset, k_seg, true);
    }
    const auto M = static_cast<int64_t>(gemm_shape.m);
    const auto N = static_cast<int64_t>(gemm_shape.n);
    const auto K = static_cast<int64_t>(gemm_shape.k);
    const auto TM = static_cast<int64_t>(gemm_shape.tile_m);
    const auto TN = static_cast<int64_t>(gemm_shape.tile_n);
    const auto KO = static_cast<int64_t>(k_offset);
    const auto KS = static_cast<int64_t>(k_seg);
    if (M <= 0 || N <= 0 || K <= 0 || TM <= 0 || TN <= 0 || KS <= 0 ||
        !extent_ok(gemm_shape.m) || !extent_ok(gemm_shape.n) || !extent_ok(gemm_shape.k)) {
        return capture_failure(TileOpStatus::invalid_argument, "gemm kseg extents out of range");
    }
    if (TM > 256 || TN > 256 || TM * TN > 16384) {
        return capture_failure(TileOpStatus::invalid_argument, "gemm kseg tile too large");
    }
    constexpr int64_t BK = kTileGemmKBlock;
    if (KO % BK != 0 || KS % BK != 0 || KO + KS > K) {
        return capture_failure(TileOpStatus::invalid_argument,
                               "gemm kseg requires block-aligned offset/length within K");
    }
    if (format == QwenTileWeightFormat::bf16 && K % 2 != 0) {
        return capture_failure(TileOpStatus::invalid_argument, "bf16 gemm requires even K");
    }
    if (format == QwenTileWeightFormat::fp8_e4m3 && K % 4 != 0) {
        return capture_failure(TileOpStatus::invalid_argument, "fp8 gemm requires K divisible by 4");
    }
    if (!volume_ok(gemm_shape.m, gemm_shape.k) || !volume_ok(gemm_shape.n, gemm_shape.k) || !volume_ok(gemm_shape.m, gemm_shape.n)) {
        return capture_failure(TileOpStatus::overflow, "gemm kseg volume overflow");
    }
    const auto groups = (K + BK - 1) / BK;
    const auto seg_blocks = KS / BK;
    const auto ko_blocks = KO / BK;
    // Partial-sum GEMM over K in [KO, KO+KS). No residual epilogue: the caller
    // accumulates segments and applies the residual after the last one. The
    // serial K loop is bounded to seg_blocks, which keeps large-K projections
    // (e.g. ffn_down K=9216) within a dispatch size the VK backend executes
    // reliably; the full-K single-dispatch variant hangs/loses the device there.
    auto definition = tile_kernel(
        format == QwenTileWeightFormat::bf16 ? "qwen_tile_gemm_bf16_kseg" : "qwen_tile_gemm_fp8_kseg",
        [=](TensorView<const float, 2> x,
            TensorView<const uint32_t, 2> w,
            TensorView<const float, 2> scales,
            TensorView<float, 2> y) {
            auto m = axis("m", TM);
            auto n = axis("n", TN);
            auto k = axis("k", BK);
            auto gm = axis("gm", ceil_div(M, TM));
            auto gn = axis("gn", ceil_div(N, TN));
            for (auto &g : parallel(shape(gm, gn))) {
                auto m0 = g.index(gm) * TM;
                auto n0 = g.index(gn) * TN;
                auto acc = zeros<float>(shape(m, n));
                for (auto &kb : serial(shape(seg_blocks))) {
                    auto k0 = KO + kb.index() * BK;
                    auto a = x.tile(coord(m0, k0), shape(m, k)).load();
                    if (format == QwenTileWeightFormat::bf16) {
                        auto k2 = axis("k2", BK / 2);
                        auto words = w.tile(coord(n0, k0 / 2), shape(n, k2)).load();
                        auto pair = decode_bf16_words(words);
                        auto a_lo = map<float>(shape(m, k2), [&](const Nest &e) {
                            return a.at(coord(e.index(m), e.index(k2) * 2));
                        });
                        auto a_hi = map<float>(shape(m, k2), [&](const Nest &e) {
                            return a.at(coord(e.index(m), e.index(k2) * 2 + 1));
                        });
                        acc = mma(a_lo, pair.lo, acc, MmaPolicy{.allow_reassociation = false});
                        acc = mma(a_hi, pair.hi, acc, MmaPolicy{.allow_reassociation = false});
                    } else {
                        auto k4 = axis("k4", BK / 4);
                        auto words = w.tile(coord(n0, k0 / 4), shape(n, k4)).load();
                        auto sc = reshape(scales.tile(coord(n0, ko_blocks + kb.index()), shape(n, axis("s1", 1))).load(), shape(n));
                        auto b0 = decode_fp8_lane(words, 0);
                        auto b1 = decode_fp8_lane(words, 1);
                        auto b2 = decode_fp8_lane(words, 2);
                        auto b3 = decode_fp8_lane(words, 3);
                        auto decoded = map<float>(shape(n, k), [&](const Nest &e) {
                            auto nn = e.index(n);
                            auto kk = e.index(k);
                            auto w4 = kk / 4;
                            auto lane = kk % 4;
                            auto v = ite(lane == 0, b0.at(coord(nn, w4)),
                                         ite(lane == 1, b1.at(coord(nn, w4)),
                                             ite(lane == 2, b2.at(coord(nn, w4)), b3.at(coord(nn, w4)))));
                            return v * sc.at(coord(nn));
                        });
                        acc = mma(a, decoded, acc, MmaPolicy{.allow_reassociation = false});
                    }
                }
                y(coord(m0, n0), shape(m, n)).store(acc);
            }
        });
    auto dummy = tensor_shape(1, 1);
    auto w_shape = format == QwenTileWeightFormat::bf16 ?
                       tensor_shape(N, K / 2) : tensor_shape(N, K / 4);
    auto s_shape = format == QwenTileWeightFormat::bf16 ? dummy : tensor_shape(N, groups);
    return finish_capture(definition.capture(
        tensor_shape(M, K), w_shape, s_shape, tensor_shape(M, N)));
}

// ---------------------------------------------------------------------------
// 4. SwiGLU.
// ---------------------------------------------------------------------------
TileCapture capture_tile_swiglu_split(uint32_t count) {
    if (!extent_ok(count)) {
        return capture_failure(TileOpStatus::invalid_argument, "swiglu count out of range");
    }
    const auto C = static_cast<int64_t>(count);
    constexpr int64_t B = 256;
    auto definition = tile_kernel("qwen_tile_swiglu_split", [=](TensorView<const float, 1> gate,
                                                                TensorView<const float, 1> up,
                                                                TensorView<float, 1> out) {
        auto e = axis("e", B);
        for (auto &block : parallel(shape(ceil_div(C, B)))) {
            auto origin = coord(block.index() * B);
            auto g = gate.tile(origin, shape(e)).load();
            auto u = up.tile(origin, shape(e)).load();
            out(origin, shape(e)).store(tile_silu(g) * u);
        }
    });
    return finish_capture(definition.capture(tensor_shape(C), tensor_shape(C), tensor_shape(C)));
}

TileCapture capture_tile_swiglu_fused(uint32_t m, uint32_t f) {
    if (!extent_ok(m) || !extent_ok(f) || !volume_ok(m, 2ull * f)) {
        return capture_failure(TileOpStatus::invalid_argument, "swiglu fused extents out of range");
    }
    const auto M = static_cast<int64_t>(m);
    const auto F = static_cast<int64_t>(f);
    constexpr int64_t FB = 256;
    auto definition = tile_kernel("qwen_tile_swiglu_fused", [=](TensorView<const float, 2> proj,
                                                                TensorView<float, 2> out) {
        auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
        auto e = axis("e", FB);
        for (auto &row : parallel(shape(M))) {
            for (auto &fb : serial(shape(ceil_div(F, FB)))) {
                auto f0 = fb.index() * FB;
                auto g = reshape(proj.tile(coord(row.index(), f0), shape(one, e)).load(), shape(e));
                auto u = reshape(proj.tile(coord(row.index(), F + f0), shape(one, e)).load(), shape(e));
                out(coord(row.index(), f0), shape(one, e)).store(reshape(tile_silu(g) * u, shape(one, e)));
            }
        }
    });
    return finish_capture(definition.capture(tensor_shape(M, 2 * F), tensor_shape(M, F)));
}

// ---------------------------------------------------------------------------
// 5. Elementwise utilities.
// ---------------------------------------------------------------------------
TileCapture capture_tile_residual_add(uint32_t count) {
    if (!extent_ok(count)) {
        return capture_failure(TileOpStatus::invalid_argument, "residual add count out of range");
    }
    const auto C = static_cast<int64_t>(count);
    constexpr int64_t B = 256;
    auto definition = tile_kernel("qwen_tile_residual_add", [=](TensorView<const float, 1> a,
                                                                TensorView<const float, 1> b,
                                                                TensorView<float, 1> out) {
        auto e = axis("e", B);
        for (auto &block : parallel(shape(ceil_div(C, B)))) {
            auto origin = coord(block.index() * B);
            out(origin, shape(e)).store(a.tile(origin, shape(e)).load() + b.tile(origin, shape(e)).load());
        }
    });
    return finish_capture(definition.capture(tensor_shape(C), tensor_shape(C), tensor_shape(C)));
}

TileCapture capture_tile_zero_fill(uint32_t count) {
    if (!extent_ok(count)) {
        return capture_failure(TileOpStatus::invalid_argument, "zero fill count out of range");
    }
    const auto C = static_cast<int64_t>(count);
    constexpr int64_t B = 256;
    auto definition = tile_kernel("qwen_tile_zero_fill", [=](TensorView<float, 1> out) {
        auto e = axis("e", B);
        for (auto &block : parallel(shape(ceil_div(C, B)))) {
            out(coord(block.index() * B), shape(e)).store(zeros<float>(shape(e)));
        }
    });
    return finish_capture(definition.capture(tensor_shape(C)));
}

// ---------------------------------------------------------------------------
// 6. Full attention.
// ---------------------------------------------------------------------------
TileCapture capture_tile_attn_prepare_q(const QwenTileConfig &cfg, uint32_t tokens) {
    if (auto v = validate_qwen_tile_config(cfg); !v.ok()) { return capture_failure(v.code, std::move(v.message)); }
    if (!extent_ok(tokens)) { return capture_failure(TileOpStatus::invalid_argument, "attn tokens out of range"); }
    const auto T = static_cast<int64_t>(tokens);
    const auto QH = static_cast<int64_t>(cfg.q_heads);
    const auto HD = static_cast<int64_t>(cfg.attn_head_dim);
    const auto eps = cfg.norm_eps;
    const auto inv_hd = 1.0f / static_cast<float>(cfg.attn_head_dim);
    if (!volume_ok(tokens, cfg.attn_q_proj_n())) {
        return capture_failure(TileOpStatus::overflow, "attn prepare q volume overflow");
    }
    // One program per (token, q_head). q_proj interleaves [Q(hd)|gate(hd)]
    // per head. Every store targets this program's own (t, h) slot: all
    // addresses are in bounds by construction (no store selects, no OOB).
    const auto q_stride = 2 * QH * HD;
    auto definition = tile_kernel("qwen_tile_attn_prepare_q", [=](TensorView<const float, 2> q_proj,
                                                                  TensorView<const float, 1> q_norm_w,
                                                                  TensorView<float, 3> q_out,
                                                                  TensorView<float, 3> gate_out) {
        auto d = axis("d", HD);
        auto t_axis = axis("t", T);
        auto h_axis = axis("h", QH);
        for (auto &g : parallel(shape(t_axis, h_axis))) {
            auto t = g.index(t_axis);
            auto h = g.index(h_axis);
            auto q_col = h * (2 * HD);
            auto q_row = serial_row(d, [&](const Scalar<int64_t> &dd) {
                return q_proj(t, q_col + dd).load();
            });
            auto q_ms = reduce(q_row * q_row, d, add, reduction::fold_left).scalar() * inv_hd;
            auto q_scale = 1.0f / sqrt(q_ms + eps);
            auto qw = q_norm_w.tile(coord(0), shape(d)).load();
            auto q_normed = map<float>(shape(d), [&](const Nest &e) {
                auto dd = e.index(d);
                return q_row.at(coord(dd)) * q_scale * qw.at(coord(dd));
            });
            for (auto &e : serial(shape(d))) {
                auto dd = e.index();
                q_out(t, h, dd).store(q_normed.at(coord(dd)));
                auto gv = q_proj(t, q_col + HD + dd).load();
                gate_out(t, h, dd).store(gv);
            }
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, q_stride), tensor_shape(HD),
        tensor_shape(T, QH, HD), tensor_shape(T, QH, HD)));
}

TileCapture capture_tile_attn_prepare_k(const QwenTileConfig &cfg, uint32_t tokens) {
    if (auto v = validate_qwen_tile_config(cfg); !v.ok()) { return capture_failure(v.code, std::move(v.message)); }
    if (!extent_ok(tokens)) { return capture_failure(TileOpStatus::invalid_argument, "attn tokens out of range"); }
    const auto T = static_cast<int64_t>(tokens);
    const auto KH = static_cast<int64_t>(cfg.kv_heads);
    const auto HD = static_cast<int64_t>(cfg.attn_head_dim);
    const auto eps = cfg.norm_eps;
    const auto inv_hd = 1.0f / static_cast<float>(cfg.attn_head_dim);
    if (!volume_ok(tokens, cfg.attn_kv_n())) {
        return capture_failure(TileOpStatus::overflow, "attn prepare k volume overflow");
    }
    // One program per (token, k_head); in-bounds stores only.
    const auto kv_stride = KH * HD;
    auto definition = tile_kernel("qwen_tile_attn_prepare_k", [=](TensorView<const float, 2> k_proj,
                                                                  TensorView<const float, 1> k_norm_w,
                                                                  TensorView<float, 3> k_out) {
        auto d = axis("d", HD);
        auto t_axis = axis("t", T);
        auto h_axis = axis("h", KH);
        for (auto &g : parallel(shape(t_axis, h_axis))) {
            auto t = g.index(t_axis);
            auto h = g.index(h_axis);
            auto k_col = h * HD;
            auto k_row = serial_row(d, [&](const Scalar<int64_t> &dd) {
                return k_proj(t, k_col + dd).load();
            });
            auto k_ms = reduce(k_row * k_row, d, add, reduction::fold_left).scalar() * inv_hd;
            auto k_scale = 1.0f / sqrt(k_ms + eps);
            auto kw = k_norm_w.tile(coord(0), shape(d)).load();
            auto k_normed = map<float>(shape(d), [&](const Nest &e) {
                auto dd = e.index(d);
                return k_row.at(coord(dd)) * k_scale * kw.at(coord(dd));
            });
            for (auto &e : serial(shape(d))) {
                auto dd = e.index();
                k_out(t, h, dd).store(k_normed.at(coord(dd)));
            }
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, kv_stride), tensor_shape(HD), tensor_shape(T, KH, HD)));
}

TileCapture capture_tile_attn_prepare_v(const QwenTileConfig &cfg, uint32_t tokens) {
    if (auto v = validate_qwen_tile_config(cfg); !v.ok()) { return capture_failure(v.code, std::move(v.message)); }
    if (!extent_ok(tokens)) { return capture_failure(TileOpStatus::invalid_argument, "attn tokens out of range"); }
    const auto T = static_cast<int64_t>(tokens);
    const auto KH = static_cast<int64_t>(cfg.kv_heads);
    const auto HD = static_cast<int64_t>(cfg.attn_head_dim);
    if (!volume_ok(tokens, cfg.attn_kv_n())) {
        return capture_failure(TileOpStatus::overflow, "attn prepare v volume overflow");
    }
    // Raw V copy, one program per (token, v_head); in-bounds stores only.
    const auto kv_stride = KH * HD;
    auto definition = tile_kernel("qwen_tile_attn_prepare_v", [=](TensorView<const float, 2> v_proj,
                                                                  TensorView<float, 3> v_out) {
        auto d = axis("d", HD);
        auto t_axis = axis("t", T);
        auto h_axis = axis("h", KH);
        for (auto &g : parallel(shape(t_axis, h_axis))) {
            auto t = g.index(t_axis);
            auto h = g.index(h_axis);
            auto v_col = h * HD;
            for (auto &e : serial(shape(d))) {
                auto dd = e.index();
                v_out(t, h, dd).store(v_proj(t, v_col + dd).load());
            }
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, kv_stride), tensor_shape(T, KH, HD)));
}

// ---------------------------------------------------------------------------
// Standalone NeoX rotate-half partial RoPE.
// ---------------------------------------------------------------------------
TileOpResult tile_host_rope_inv_freq(uint32_t rope_dim, float theta, luisa::vector<float> &inv_freq) {
    if (rope_dim == 0 || rope_dim % 2 != 0) { return invalid("rope_dim must be even and > 0"); }
    if (!(theta > 0.0f) || !std::isfinite(theta)) { return invalid("theta must be finite > 0"); }
    inv_freq.resize(rope_dim / 2);
    for (auto i = 0u; i < rope_dim / 2; i++) {
        inv_freq[i] = static_cast<float>(std::pow(static_cast<double>(theta),
                                                  -2.0 * static_cast<double>(i) / static_cast<double>(rope_dim)));
    }
    return TileOpResult::success();
}

TileOpResult tile_host_rope_tables(uint32_t rope_dim, float theta, uint32_t max_context,
                                   luisa::vector<float> &cos_tab, luisa::vector<float> &sin_tab) {
    if (max_context == 0) { return invalid("rope table max_context must be > 0"); }
    luisa::vector<float> inv_freq;
    if (auto r = tile_host_rope_inv_freq(rope_dim, theta, inv_freq); !r.ok()) { return r; }
    const auto pairs = inv_freq.size();
    if (!volume_ok(max_context, pairs)) { return invalid("rope table volume overflow"); }
    cos_tab.resize(static_cast<size_t>(max_context) * pairs);
    sin_tab.resize(static_cast<size_t>(max_context) * pairs);
    for (uint32_t pos = 0; pos < max_context; pos++) {
        for (size_t i = 0; i < pairs; i++) {
            // Double-precision angle and libm evaluation; single FP32 rounding.
            const auto angle = static_cast<double>(pos) * static_cast<double>(inv_freq[i]);
            const auto idx = static_cast<size_t>(pos) * pairs + i;
            cos_tab[idx] = static_cast<float>(std::cos(angle));
            sin_tab[idx] = static_cast<float>(std::sin(angle));
        }
    }
    return TileOpResult::success();
}

TileCapture capture_tile_rope_partial_table(uint32_t tokens, uint32_t heads, uint32_t head_dim,
                                            uint32_t rope_dim, uint32_t max_context) {
    if (!extent_ok(tokens) || !extent_ok(heads) || !extent_ok(head_dim) || !extent_ok(rope_dim) ||
        !extent_ok(max_context)) {
        return capture_failure(TileOpStatus::invalid_argument, "rope table extents out of range");
    }
    if (rope_dim % 2 != 0 || rope_dim > head_dim) {
        return capture_failure(TileOpStatus::invalid_argument, "rope_dim must be even and <= head_dim");
    }
    if (!volume_ok(tokens, heads, head_dim) || !volume_ok(max_context, rope_dim / 2)) {
        return capture_failure(TileOpStatus::overflow, "rope table volume overflow");
    }
    const auto T = static_cast<int64_t>(tokens);
    const auto H = static_cast<int64_t>(heads);
    const auto HD = static_cast<int64_t>(head_dim);
    const auto RD = static_cast<int64_t>(rope_dim);
    const auto MC = static_cast<int64_t>(max_context);
    const auto pairs = RD / 2;
    // Interleaved partial RoPE from host-precomputed FP32 sin/cos tables
    // indexed by ABSOLUTE position. No device sin/cos evaluation: the table
    // entries are exact libm values rounded once to FP32 on the host.
    auto definition = tile_kernel("qwen_tile_rope_partial_table", [=](TensorView<const float, 3> x,
                                                                      TensorView<const int32_t, 1> positions,
                                                                      TensorView<const float, 2> cos_tab,
                                                                      TensorView<const float, 2> sin_tab,
                                                                      TensorView<float, 3> out) {
        auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
        auto d = axis("d", HD);
        auto t_axis = axis("t", T);
        auto h_axis = axis("h", H);
        for (auto &g : parallel(shape(t_axis, h_axis))) {
            auto t = g.index(t_axis);
            auto h = g.index(h_axis);
            auto row = reshape(x.tile(coord(t, h, 0), shape(one, one2, d)).load(), shape(d));
            auto pos = cast<int64_t>(positions(t).load());
            // Table rows are loaded in the execution nest (map is pure).
            // The caller guarantees 0 <= pos < max_context; the row index is
            // clamped defensively so a bad position can never read OOB.
            auto pos_row = min(max(pos, Scalar<int64_t>{0}), Scalar<int64_t>{MC - 1});
            auto cos_row = zeros<float>(shape(d));
            auto sin_row = zeros<float>(shape(d));
            for (auto &e : serial(shape(d))) {
                auto dd = e.index();
                // NeoX rotate-half: element dd pairs with dd +/- pairs, and
                // both members of a pair share frequency index dd % pairs.
                auto pair = dd % pairs;
                auto cv = cos_tab(pos_row, pair).load();
                auto sv = sin_tab(pos_row, pair).load();
                cos_row = map<float>(shape(d), [&](const Nest &m) {
                    return ite(m.index(d) == dd, cv, cos_row.at(coord(m.index(d))));
                });
                sin_row = map<float>(shape(d), [&](const Nest &m) {
                    return ite(m.index(d) == dd, sv, sin_row.at(coord(m.index(d))));
                });
            }
            auto roped = map<float>(shape(d), [&](const Nest &e) {
                auto dd = e.index(d);
                // NeoX rotate-half pairs (i, i+pairs) for i < pairs; the
                // tail [RD, HD) passes through unchanged.
                auto lo = dd < pairs;
                auto partner = ite(lo, dd + pairs, dd - pairs);
                auto cv = cos_row.at(coord(dd));
                auto sv = sin_row.at(coord(dd));
                auto self = row.at(coord(dd));
                auto other = row.at(coord(partner));
                // lo: x0*cos - x1*sin; hi: x0*sin + x1*cos
                auto rotated = ite(lo, self * cv - other * sv, other * sv + self * cv);
                return ite(dd < RD, rotated, self);
            });
            out(coord(t, h, 0), shape(one, one2, d)).store(reshape(roped, shape(one, one2, d)));
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, H, HD), tensor_shape(T), tensor_shape(MC, pairs), tensor_shape(MC, pairs),
        tensor_shape(T, H, HD)));
}

TileCapture capture_tile_rope_partial(uint32_t tokens, uint32_t heads, uint32_t head_dim,
                                      uint32_t rope_dim) {
    if (!extent_ok(tokens) || !extent_ok(heads) || !extent_ok(head_dim) || !extent_ok(rope_dim)) {
        return capture_failure(TileOpStatus::invalid_argument, "rope extents out of range");
    }
    if (rope_dim % 2 != 0 || rope_dim > head_dim) {
        return capture_failure(TileOpStatus::invalid_argument, "rope_dim must be even and <= head_dim");
    }
    if (!volume_ok(tokens, heads, head_dim)) {
        return capture_failure(TileOpStatus::overflow, "rope volume overflow");
    }
    const auto T = static_cast<int64_t>(tokens);
    const auto H = static_cast<int64_t>(heads);
    const auto HD = static_cast<int64_t>(head_dim);
    const auto RD = static_cast<int64_t>(rope_dim);
    const auto pairs = RD / 2;
    auto definition = tile_kernel("qwen_tile_rope_partial", [=](TensorView<const float, 3> x,
                                                                TensorView<const int32_t, 1> positions,
                                                                TensorView<const float, 1> inv_freq_view,
                                                                TensorView<float, 3> out) {
        auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
        auto d = axis("d", HD);
        auto t_axis = axis("t", T);
        auto h_axis = axis("h", H);
        for (auto &g : parallel(shape(t_axis, h_axis))) {
            auto t = g.index(t_axis);
            auto h = g.index(h_axis);
            auto row = reshape(x.tile(coord(t, h, 0), shape(one, one2, d)).load(), shape(d));
            auto pos = cast<float>(positions(t).load());
            // Frequencies are loaded in the execution nest (map is pure).
            auto freqs = zeros<float>(shape(d));
            for (auto &e : serial(shape(d))) {
                auto dd = e.index();
                // NeoX rotate-half: frequency index dd % pairs (both pair
                // members share it); the tail index is clamped in bounds.
                auto pair = dd % pairs;
                auto fv = inv_freq_view(pair).load();
                freqs = map<float>(shape(d), [&](const Nest &m) {
                    return ite(m.index(d) == dd, fv, freqs.at(coord(m.index(d))));
                });
            }
            auto roped = map<float>(shape(d), [&](const Nest &e) {
                auto dd = e.index(d);
                // NeoX rotate-half pairs (i, i+pairs); tail passes through.
                auto lo = dd < pairs;
                auto partner = ite(lo, dd + pairs, dd - pairs);
                auto sc = scalar_sin_cos(pos * freqs.at(coord(dd)));
                auto self = row.at(coord(dd));
                auto other = row.at(coord(partner));
                // lo: x0*cos - x1*sin; hi: x0*sin + x1*cos
                auto rotated = ite(lo, self * sc.cos_v - other * sc.sin_v,
                                   other * sc.sin_v + self * sc.cos_v);
                return ite(dd < RD, rotated, self);
            });
            out(coord(t, h, 0), shape(one, one2, d)).store(reshape(roped, shape(one, one2, d)));
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, H, HD), tensor_shape(T), tensor_shape(pairs), tensor_shape(T, H, HD)));
}

// ---------------------------------------------------------------------------
// KV append.
// ---------------------------------------------------------------------------
TileCapture capture_tile_kv_append(uint32_t tokens, uint32_t channels, uint32_t pos0, uint32_t capacity) {
    if (!extent_ok(tokens) || !extent_ok(channels) || !extent_ok(capacity) ||
        !volume_ok(tokens, channels) || !volume_ok(capacity, channels)) {
        return capture_failure(TileOpStatus::invalid_argument, "kv append extents out of range");
    }
    if (static_cast<uint64_t>(pos0) + tokens > capacity) {
        return capture_failure(TileOpStatus::invalid_argument, "kv append exceeds cache capacity");
    }
    const auto T = static_cast<int64_t>(tokens);
    const auto C = static_cast<int64_t>(channels);
    const auto P0 = static_cast<int64_t>(pos0);
    const auto CAP = static_cast<int64_t>(capacity);
    constexpr int64_t CB = 256;
    auto definition = tile_kernel("qwen_tile_kv_append", [=](TensorView<const float, 2> k,
                                                             TensorView<const float, 2> v,
                                                             TensorView<float, 2> k_cache,
                                                             TensorView<float, 2> v_cache) {
        auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
        auto c = axis("c", CB);
        for (auto &t : parallel(shape(T))) {
            for (auto &cb : serial(shape(ceil_div(C, CB)))) {
                auto c0 = cb.index() * CB;
                auto kv = k.tile(coord(t.index(), c0), shape(one, c)).load();
                auto vv = v.tile(coord(t.index(), c0), shape(one, c)).load();
                k_cache(coord(P0 + t.index(), c0), shape(one, c)).store(kv);
                v_cache(coord(P0 + t.index(), c0), shape(one, c)).store(vv);
            }
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, C), tensor_shape(T, C), tensor_shape(CAP, C), tensor_shape(CAP, C)));
}

// ---------------------------------------------------------------------------
// Causal GQA prefill attention (online FP32 softmax + sigmoid output gate).
// ---------------------------------------------------------------------------
namespace {

// Shared attention body: one program per (query row, q_head). `q_index(t)`
// maps the program's token index to the absolute cache diagonal.
struct AttnDims {
    int64_t q_heads, kv_heads, head_dim, group;
    float scale;
};

}// namespace

TileCapture capture_tile_attn_prefill(const QwenTileConfig &cfg, uint32_t tokens,
                                      uint32_t pos0, uint32_t kv_len, uint32_t capacity) {
    if (auto v = validate_qwen_tile_config(cfg); !v.ok()) { return capture_failure(v.code, std::move(v.message)); }
    if (!extent_ok(tokens) || !extent_ok(capacity) || !extent_ok(kv_len)) {
        return capture_failure(TileOpStatus::invalid_argument, "attn prefill extents out of range");
    }
    if (kv_len == 0 || static_cast<uint64_t>(pos0) + tokens > kv_len || kv_len > capacity) {
        return capture_failure(TileOpStatus::invalid_argument, "attn prefill requires pos0+tokens <= kv_len <= capacity");
    }
    const auto T = static_cast<int64_t>(tokens);
    const auto QH = static_cast<int64_t>(cfg.q_heads);
    const auto KH = static_cast<int64_t>(cfg.kv_heads);
    const auto HD = static_cast<int64_t>(cfg.attn_head_dim);
    const auto CAP = static_cast<int64_t>(capacity);
    const auto KV = static_cast<int64_t>(kv_len);
    const auto P0 = static_cast<int64_t>(pos0);
    const auto group = QH / KH;
    const auto scale = cfg.attn_scale();
    auto definition = tile_kernel("qwen_tile_attn_prefill", [=](TensorView<const float, 3> q,
                                                                TensorView<const float, 3> gate,
                                                                TensorView<const float, 3> k_cache,
                                                                TensorView<const float, 3> v_cache,
                                                                TensorView<float, 3> out) {
        auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
        auto d = axis("d", HD);
        auto t_axis = axis("t", T);
        auto h_axis = axis("h", QH);
        for (auto &g : parallel(shape(t_axis, h_axis))) {
            auto t = g.index(t_axis);
            auto h = g.index(h_axis);
            auto kv_head = h / group;
            auto qv = reshape(q.tile(coord(t, h, 0), shape(one, one2, d)).load(), shape(d));
            auto limit = P0 + t + 1;// causal: keys [0, pos0+t]
            auto peak = Scalar<float>{-3.0e38f};
            auto sum = Scalar<float>{0.0f};
            auto acc = zeros<float>(shape(d));
            for (auto &j : serial(shape(KV))) {
                auto active = j.index() < limit;
                auto kv = reshape(k_cache.tile(coord(j.index(), kv_head, 0), shape(one, one2, d)).load(), shape(d));
                auto score = ite(active, reduce(qv * kv, d, add, reduction::fold_left).scalar() * scale, -3.0e38f);
                auto new_peak = max(peak, score);
                auto correction = exp(peak - new_peak);
                auto weight = ite(active, exp(score - new_peak), 0.0f);
                auto vv = reshape(v_cache.tile(coord(j.index(), kv_head, 0), shape(one, one2, d)).load(), shape(d));
                acc = acc * correction + vv * weight;
                sum = sum * correction + weight;
                peak = new_peak;
            }
            auto result = acc / sum;
            auto gv = reshape(gate.tile(coord(t, h, 0), shape(one, one2, d)).load(), shape(d));
            out(coord(t, h, 0), shape(one, one2, d)).store(reshape(result * tile_sigmoid(gv), shape(one, one2, d)));
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, QH, HD), tensor_shape(T, QH, HD),
        tensor_shape(CAP, KH, HD), tensor_shape(CAP, KH, HD), tensor_shape(T, QH, HD)));
}

// ---------------------------------------------------------------------------
// Device-visible-length attention (integration extension for the executor).
// The static kv_len captures above specialize the serial key loop to one
// exact length; the dynamic variants below stream the FULL capacity and gate
// each key with a device-visible limit, so one compiled shader serves every
// position (no per-token JIT). The scalar online-softmax carries and the
// serial-nest ite(active, ...) selects are the verified-safe pattern (scalar
// bool conditions that vary per serial iteration are NOT the uniform
// tile-lane select that miscompiles on the XIR route).
// ---------------------------------------------------------------------------
TileCapture capture_tile_attn_prefill_dynamic(const QwenTileConfig &cfg, uint32_t tokens,
                                              uint32_t capacity) {
    if (auto v = validate_qwen_tile_config(cfg); !v.ok()) { return capture_failure(v.code, std::move(v.message)); }
    if (!extent_ok(tokens) || !extent_ok(capacity) || static_cast<uint64_t>(tokens) > capacity) {
        return capture_failure(TileOpStatus::invalid_argument, "attn prefill dynamic requires 1 <= tokens <= capacity");
    }
    const auto T = static_cast<int64_t>(tokens);
    const auto QH = static_cast<int64_t>(cfg.q_heads);
    const auto KH = static_cast<int64_t>(cfg.kv_heads);
    const auto HD = static_cast<int64_t>(cfg.attn_head_dim);
    const auto CAP = static_cast<int64_t>(capacity);
    const auto group = QH / KH;
    const auto scale = cfg.attn_scale();
    auto definition = tile_kernel("qwen_tile_attn_prefill_dyn", [=](TensorView<const float, 3> q,
                                                                    TensorView<const float, 3> gate,
                                                                    TensorView<const float, 3> k_cache,
                                                                    TensorView<const float, 3> v_cache,
                                                                    TensorView<const int32_t, 1> pos0_len,
                                                                    TensorView<float, 3> out) {
        auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
        auto d = axis("d", HD);
        auto t_axis = axis("t", T);
        auto h_axis = axis("h", QH);
        for (auto &g : parallel(shape(t_axis, h_axis))) {
            auto t = g.index(t_axis);
            auto h = g.index(h_axis);
            auto kv_head = h / group;
            // pos0_len = [pos0, kv_len]; the caller guarantees
            // 1 <= kv_len <= capacity and pos0 + tokens <= kv_len.
            auto pos0 = cast<int64_t>(pos0_len(Scalar<int64_t>{0}).load());
            auto kv_len = cast<int64_t>(pos0_len(Scalar<int64_t>{1}).load());
            auto limit = min(pos0 + t + 1, kv_len);// causal diagonal, clamped
            auto qv = reshape(q.tile(coord(t, h, 0), shape(one, one2, d)).load(), shape(d));
            auto peak = Scalar<float>{-3.0e38f};
            auto sum = Scalar<float>{0.0f};
            auto acc = zeros<float>(shape(d));
            for (auto &j : serial(shape(CAP))) {
                auto active = j.index() < limit;
                auto kv = reshape(k_cache.tile(coord(j.index(), kv_head, 0), shape(one, one2, d)).load(), shape(d));
                auto score = ite(active, reduce(qv * kv, d, add, reduction::fold_left).scalar() * scale, Scalar<float>{-3.0e38f});
                auto new_peak = max(peak, score);
                auto correction = exp(peak - new_peak);
                auto weight = ite(active, exp(score - new_peak), Scalar<float>{0.0f});
                auto vv = reshape(v_cache.tile(coord(j.index(), kv_head, 0), shape(one, one2, d)).load(), shape(d));
                acc = acc * correction + vv * weight;
                sum = sum * correction + weight;
                peak = new_peak;
            }
            auto result = acc / sum;
            auto gv = reshape(gate.tile(coord(t, h, 0), shape(one, one2, d)).load(), shape(d));
            out(coord(t, h, 0), shape(one, one2, d)).store(reshape(result * tile_sigmoid(gv), shape(one, one2, d)));
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, QH, HD), tensor_shape(T, QH, HD),
        tensor_shape(CAP, KH, HD), tensor_shape(CAP, KH, HD),
        tensor_shape(2), tensor_shape(T, QH, HD)));
}

TileCapture capture_tile_attn_decode_dynamic(const QwenTileConfig &cfg, uint32_t capacity) {
    if (auto v = validate_qwen_tile_config(cfg); !v.ok()) { return capture_failure(v.code, std::move(v.message)); }
    if (!extent_ok(capacity)) {
        return capture_failure(TileOpStatus::invalid_argument, "attn decode dynamic capacity out of range");
    }
    const auto QH = static_cast<int64_t>(cfg.q_heads);
    const auto KH = static_cast<int64_t>(cfg.kv_heads);
    const auto HD = static_cast<int64_t>(cfg.attn_head_dim);
    const auto CAP = static_cast<int64_t>(capacity);
    const auto group = QH / KH;
    const auto scale = cfg.attn_scale();
    auto definition = tile_kernel("qwen_tile_attn_decode_dyn", [=](TensorView<const float, 2> q,
                                                                   TensorView<const float, 2> gate,
                                                                   TensorView<const float, 3> k_cache,
                                                                   TensorView<const float, 3> v_cache,
                                                                   TensorView<const int32_t, 1> kv_len_buf,
                                                                   TensorView<float, 2> out) {
        auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
        auto d = axis("d", HD);
        for (auto &h : parallel(shape(QH))) {
            auto kv_head = h.index() / group;
            auto kv_len = cast<int64_t>(kv_len_buf(Scalar<int64_t>{0}).load());
            auto qv = reshape(q.tile(coord(h.index(), 0), shape(one, d)).load(), shape(d));
            auto peak = Scalar<float>{-3.0e38f};
            auto sum = Scalar<float>{0.0f};
            auto acc = zeros<float>(shape(d));
            for (auto &j : serial(shape(CAP))) {
                auto active = j.index() < kv_len;
                auto kv = reshape(k_cache.tile(coord(j.index(), kv_head, 0), shape(one, one2, d)).load(), shape(d));
                auto score = ite(active, reduce(qv * kv, d, add, reduction::fold_left).scalar() * scale, Scalar<float>{-3.0e38f});
                auto new_peak = max(peak, score);
                auto correction = exp(peak - new_peak);
                auto weight = ite(active, exp(score - new_peak), Scalar<float>{0.0f});
                auto vv = reshape(v_cache.tile(coord(j.index(), kv_head, 0), shape(one, one2, d)).load(), shape(d));
                acc = acc * correction + vv * weight;
                sum = sum * correction + weight;
                peak = new_peak;
            }
            auto result = acc / sum;
            auto gv = reshape(gate.tile(coord(h.index(), 0), shape(one, d)).load(), shape(d));
            out(coord(h.index(), 0), shape(one, d)).store(reshape(result * tile_sigmoid(gv), shape(one, d)));
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(QH, HD), tensor_shape(QH, HD),
        tensor_shape(CAP, KH, HD), tensor_shape(CAP, KH, HD),
        tensor_shape(1), tensor_shape(QH, HD)));
}

// ---------------------------------------------------------------------------
// Single-query decode attention.
// ---------------------------------------------------------------------------
TileCapture capture_tile_attn_decode(const QwenTileConfig &cfg, uint32_t kv_len, uint32_t capacity) {
    if (auto v = validate_qwen_tile_config(cfg); !v.ok()) { return capture_failure(v.code, std::move(v.message)); }
    if (!extent_ok(kv_len) || !extent_ok(capacity) || kv_len == 0 || kv_len > capacity) {
        return capture_failure(TileOpStatus::invalid_argument, "attn decode requires 1 <= kv_len <= capacity");
    }
    const auto QH = static_cast<int64_t>(cfg.q_heads);
    const auto KH = static_cast<int64_t>(cfg.kv_heads);
    const auto HD = static_cast<int64_t>(cfg.attn_head_dim);
    const auto CAP = static_cast<int64_t>(capacity);
    const auto KV = static_cast<int64_t>(kv_len);
    const auto group = QH / KH;
    const auto scale = cfg.attn_scale();
    auto definition = tile_kernel("qwen_tile_attn_decode", [=](TensorView<const float, 2> q,
                                                               TensorView<const float, 2> gate,
                                                               TensorView<const float, 3> k_cache,
                                                               TensorView<const float, 3> v_cache,
                                                               TensorView<float, 2> out) {
        auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
        auto d = axis("d", HD);
        for (auto &h : parallel(shape(QH))) {
            auto kv_head = h.index() / group;
            auto qv = reshape(q.tile(coord(h.index(), 0), shape(one, d)).load(), shape(d));
            auto peak = Scalar<float>{-3.0e38f};
            auto sum = Scalar<float>{0.0f};
            auto acc = zeros<float>(shape(d));
            for (auto &j : serial(shape(KV))) {
                auto kv = reshape(k_cache.tile(coord(j.index(), kv_head, 0), shape(one, one2, d)).load(), shape(d));
                auto score = reduce(qv * kv, d, add, reduction::fold_left).scalar() * scale;
                auto new_peak = max(peak, score);
                auto correction = exp(peak - new_peak);
                auto weight = exp(score - new_peak);
                auto vv = reshape(v_cache.tile(coord(j.index(), kv_head, 0), shape(one, one2, d)).load(), shape(d));
                acc = acc * correction + vv * weight;
                sum = sum * correction + weight;
                peak = new_peak;
            }
            auto result = acc / sum;
            auto gv = reshape(gate.tile(coord(h.index(), 0), shape(one, d)).load(), shape(d));
            out(coord(h.index(), 0), shape(one, d)).store(reshape(result * tile_sigmoid(gv), shape(one, d)));
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(QH, HD), tensor_shape(QH, HD),
        tensor_shape(CAP, KH, HD), tensor_shape(CAP, KH, HD), tensor_shape(QH, HD)));
}

// ---------------------------------------------------------------------------
// 7. Gated DeltaNet.
// ---------------------------------------------------------------------------
TileCapture capture_tile_gdn_conv_silu_step(uint32_t channels, uint32_t kernel_size,
                                            uint32_t pos_mod) {
    if (!extent_ok(channels) || !extent_ok(kernel_size) || kernel_size < 2 ||
        !volume_ok(kernel_size, channels)) {
        return capture_failure(TileOpStatus::invalid_argument, "gdn conv step extents out of range");
    }
    const auto C = static_cast<int64_t>(channels);
    const auto KS = static_cast<int64_t>(kernel_size);
    const auto R = KS - 1;// ring slots
    if (pos_mod >= static_cast<uint32_t>(R)) {
        return capture_failure(TileOpStatus::invalid_argument, "gdn conv step pos_mod out of range");
    }
    const auto PM = static_cast<int64_t>(pos_mod);
    if (!volume_ok(channels, static_cast<uint64_t>(R))) {
        return capture_failure(TileOpStatus::overflow, "gdn conv step ring volume overflow");
    }
    constexpr int64_t CB = 64;
    // One-token conv+SiLU with SEPARATE ring_in (read-only) and ring_out
    // (write-only). ring_out copies every history slot unchanged except slot
    // PM, which receives the current PRE-activation source row. The caller
    // ping-pongs two rings per layer. Tap j < R reads history position
    // pos-R+j from ring_in slot (PM+j)%R (a compile-time constant slot, so
    // the load address is select-free); the tap is masked to 0 when the
    // absolute position pos-R+j is negative. Tap j == R reads src. The
    // absolute pos comes from a device buffer so pos=0 and pos=R (same PM)
    // are distinguished exactly. All loads/stores are in bounds.
    auto definition = tile_kernel("qwen_tile_gdn_conv_silu_step", [=](TensorView<const float, 2> src,
                                                                      TensorView<const float, 2> weight,
                                                                      TensorView<const float, 2> ring_in,
                                                                      TensorView<const int32_t, 1> pos_buf,
                                                                      TensorView<float, 2> ring_out,
                                                                      TensorView<float, 2> out) {
        auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
        auto c = axis("c", CB);
        auto r = axis("r", R);
        for (auto &block : parallel(shape(ceil_div(C, CB)))) {
            auto c0 = block.index() * CB;
            auto pos = cast<int64_t>(pos_buf(Scalar<int64_t>{0}).load());
            auto history = ring_in.tile(coord(c0, 0), shape(c, r)).load();
            auto acc = zeros<float>(shape(c));
            for (auto &tap : serial(shape(KS))) {
                auto j = tap.index();
                // History taps read ring_in directly with SCALAR loads in a
                // serial_row (the verified-safe pattern for tap-uniform slot
                // indices). Slot (PM+j)%R is a compile-time constant per tap;
                // negative absolute positions contribute 0 via an arithmetic
                // mask. The current tap (j == R) reads src instead.
                auto slot = (PM + j) % R;
                auto hist_pos = pos - R + j;
                auto hist_mask = cast<float>((hist_pos >= 0) && (j < R));
                auto cur_mask = cast<float>(j == R);
                auto tap_value = serial_row(c, [&](const Scalar<int64_t> &cc) {
                    auto hv = ring_in(c0 + cc, slot).load() * hist_mask;
                    auto sv = src(Scalar<int64_t>{0}, c0 + cc).load() * cur_mask;
                    // GGUF ssm_conv1d [kernel, channels] payload is
                    // CHANNEL-major: weight[c*kernel + j] is tap j of channel c.
                    auto wv = weight(c0 + cc, j).load();
                    return (hv + sv) * wv;
                });
                acc = acc + tap_value;
            }
            out(coord(0, c0), shape(one, c)).store(reshape(tile_silu(acc), shape(one, c)));
            // ring_out: copy history unchanged except slot PM <- current.
            // Load the current row as a tile, then select per (c, r) lane
            // (the predicate varies across the r lanes: verified-safe pure
            // map select, not a uniform store select).
            auto current = reshape(src.tile(coord(0, c0), shape(one, c)).load(), shape(c));
            auto next_history = map<float>(shape(c, r), [&](const Nest &e) {
                auto cc = e.index(c);
                auto ss = e.index(r);
                return ite(ss == PM, current.at(coord(cc)), history.at(coord(cc, ss)));
            });
            ring_out(coord(c0, 0), shape(c, r)).store(next_history);
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(1, C), tensor_shape(C, KS), tensor_shape(C, R), tensor_shape(1),
        tensor_shape(C, R), tensor_shape(1, C)));
}

TileCapture capture_tile_gdn_gates(uint32_t tokens, uint32_t heads, float softplus_threshold) {
    if (!extent_ok(tokens) || !extent_ok(heads) || !volume_ok(tokens, heads) ||
        !(softplus_threshold > 0.0f)) {
        return capture_failure(TileOpStatus::invalid_argument, "gdn gates extents out of range");
    }
    const auto T = static_cast<int64_t>(tokens);
    const auto H = static_cast<int64_t>(heads);
    auto definition = tile_kernel("qwen_tile_gdn_gates", [=](TensorView<const float, 2> a,
                                                             TensorView<const float, 2> b,
                                                             TensorView<const float, 1> ssm_a,
                                                             TensorView<const float, 1> dt_bias,
                                                             TensorView<float, 2> beta_out,
                                                             TensorView<float, 2> g_out) {
        auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
        auto h = axis("h", H);
        for (auto &t : parallel(shape(T))) {
            auto av = reshape(a.tile(coord(t.index(), 0), shape(one, h)).load(), shape(h));
            auto bv = reshape(b.tile(coord(t.index(), 0), shape(one, h)).load(), shape(h));
            auto beta = tile_sigmoid(bv);
            auto g = serial_row(h, [&](const Scalar<int64_t> &hh) {
                // GGUF ssm_a is ALREADY -exp(A_log) (negative, folded offline
                // by the converter): g = ssm_a[h] * softplus(a + dt_bias[h]).
                auto sp = scalar_softplus(av.at(coord(hh)) + dt_bias(hh).load(), softplus_threshold);
                return ssm_a(hh).load() * sp;
            });
            beta_out(coord(t.index(), 0), shape(one, h)).store(reshape(beta, shape(one, h)));
            g_out(coord(t.index(), 0), shape(one, h)).store(reshape(g, shape(one, h)));
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, H), tensor_shape(T, H), tensor_shape(H), tensor_shape(H),
        tensor_shape(T, H), tensor_shape(T, H)));
}

TileCapture capture_tile_gdn_q_l2norm(uint32_t tokens, uint32_t qk_heads, uint32_t head_dim,
                                      uint32_t v_channels, float eps, float q_scale) {
    if (!extent_ok(tokens) || !extent_ok(qk_heads) || !extent_ok(head_dim) ||
        !extent_ok(v_channels, true) || !(eps > 0.0f) || !(q_scale > 0.0f)) {
        return capture_failure(TileOpStatus::invalid_argument, "gdn q norm extents out of range");
    }
    const auto T = static_cast<int64_t>(tokens);
    const auto QK = static_cast<int64_t>(qk_heads);
    const auto HD = static_cast<int64_t>(head_dim);
    const auto VC = static_cast<int64_t>(v_channels);
    const auto stride = 2 * QK * HD + VC;
    if (!volume_ok(tokens, static_cast<uint64_t>(stride))) {
        return capture_failure(TileOpStatus::overflow, "gdn q norm volume overflow");
    }
    // One program per (token, qk_head); reads the Q section (column 0) of the
    // packed conv output and writes ONLY q_out at its own (t, h) slot. All
    // stores are in bounds; no store selects, no out-of-bounds tricks.
    auto definition = tile_kernel("qwen_tile_gdn_q_l2norm", [=](TensorView<const float, 2> conv_out,
                                                                TensorView<float, 3> q_out) {
        auto d = axis("d", HD);
        auto t_axis = axis("t", T);
        auto h_axis = axis("h", QK);
        for (auto &g : parallel(shape(t_axis, h_axis))) {
            auto t = g.index(t_axis);
            auto h = g.index(h_axis);
            auto col = h * HD;
            auto row = serial_row(d, [&](const Scalar<int64_t> &dd) {
                return conv_out(t, col + dd).load();
            });
            // FLA/HF l2norm: x * rsqrt(sum(x*x) + eps) -- eps INSIDE the
            // rsqrt (near-zero rows stay near zero, never blown up to 1/eps).
            auto ss = reduce(row * row, d, add, reduction::fold_left).scalar();
            auto inv = 1.0f / sqrt(ss + eps);
            auto scaled = row * (inv * q_scale);
            for (auto &e : serial(shape(d))) {
                auto dd = e.index();
                q_out(t, h, dd).store(scaled.at(coord(dd)));
            }
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, stride), tensor_shape(T, QK, HD)));
}

TileCapture capture_tile_gdn_k_l2norm(uint32_t tokens, uint32_t qk_heads, uint32_t head_dim,
                                      uint32_t v_channels, float eps) {
    if (!extent_ok(tokens) || !extent_ok(qk_heads) || !extent_ok(head_dim) ||
        !extent_ok(v_channels, true) || !(eps > 0.0f)) {
        return capture_failure(TileOpStatus::invalid_argument, "gdn k norm extents out of range");
    }
    const auto T = static_cast<int64_t>(tokens);
    const auto QK = static_cast<int64_t>(qk_heads);
    const auto HD = static_cast<int64_t>(head_dim);
    const auto VC = static_cast<int64_t>(v_channels);
    const auto stride = 2 * QK * HD + VC;
    if (!volume_ok(tokens, static_cast<uint64_t>(stride))) {
        return capture_failure(TileOpStatus::overflow, "gdn k norm volume overflow");
    }
    // One program per (token, qk_head); reads the K section (column QK*HD)
    // and writes ONLY k_out at its own (t, h) slot. In-bounds stores only.
    auto definition = tile_kernel("qwen_tile_gdn_k_l2norm", [=](TensorView<const float, 2> conv_out,
                                                                TensorView<float, 3> k_out) {
        auto d = axis("d", HD);
        auto t_axis = axis("t", T);
        auto h_axis = axis("h", QK);
        for (auto &g : parallel(shape(t_axis, h_axis))) {
            auto t = g.index(t_axis);
            auto h = g.index(h_axis);
            auto col = QK * HD + h * HD;
            auto row = serial_row(d, [&](const Scalar<int64_t> &dd) {
                return conv_out(t, col + dd).load();
            });
            // FLA/HF l2norm: eps inside the rsqrt (see q variant).
            auto ss = reduce(row * row, d, add, reduction::fold_left).scalar();
            auto inv = 1.0f / sqrt(ss + eps);
            auto scaled = row * inv;
            for (auto &e : serial(shape(d))) {
                auto dd = e.index();
                k_out(t, h, dd).store(scaled.at(coord(dd)));
            }
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, stride), tensor_shape(T, QK, HD)));
}

TileCapture capture_tile_gdn_recurrent(uint32_t tokens, uint32_t v_heads,
                                       uint32_t qk_heads, uint32_t head_dim) {
    if (!extent_ok(tokens) || !extent_ok(v_heads) || !extent_ok(qk_heads) || !extent_ok(head_dim) ||
        v_heads % qk_heads != 0) {
        return capture_failure(TileOpStatus::invalid_argument, "gdn recurrent extents out of range");
    }
    if (head_dim > 256 || head_dim * head_dim > 65536) {
        return capture_failure(TileOpStatus::invalid_argument, "gdn state tile too large (hd <= 256)");
    }
    const auto T = static_cast<int64_t>(tokens);
    const auto VH = static_cast<int64_t>(v_heads);
    const auto QK = static_cast<int64_t>(qk_heads);
    const auto HD = static_cast<int64_t>(head_dim);
    const auto v_offset = 2 * QK * HD;
    const auto qkv_stride = v_offset + VH * HD;
    if (!volume_ok(tokens, static_cast<uint64_t>(qkv_stride)) || !volume_ok(v_heads, head_dim, head_dim)) {
        return capture_failure(TileOpStatus::overflow, "gdn recurrent volume overflow");
    }
    // One program per V head; the [hd,hd] state tile is an SSA carry through
    // the serial token loop. Delta rule (exact per-token recurrence):
    //   S <- exp(g)*S; err = v - S^T k; S += beta * k err^T; o = S^T q.
    auto definition = tile_kernel("qwen_tile_gdn_recurrent", [=](TensorView<const float, 3> q,
                                                                 TensorView<const float, 3> k,
                                                                 TensorView<const float, 2> conv_out,
                                                                 TensorView<const float, 2> beta,
                                                                 TensorView<const float, 2> g,
                                                                 TensorView<float, 3> state,
                                                                 TensorView<float, 3> out) {
        auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
        auto kk = axis("kk", HD);
        auto vv = axis("vv", HD);
        for (auto &h : parallel(shape(VH))) {
            auto qk_head = h.index() % QK;// ggml tiled broadcast
            auto s = reshape(state.tile(coord(h.index(), 0, 0), shape(one, kk, vv)).load(), shape(kk, vv));
            for (auto &t : serial(shape(T))) {
                auto qv = reshape(q.tile(coord(t.index(), qk_head, 0), shape(one, one2, kk)).load(), shape(kk));
                auto kv = reshape(k.tile(coord(t.index(), qk_head, 0), shape(one, one2, kk)).load(), shape(kk));
                auto vvec = serial_row(vv, [&](const Scalar<int64_t> &d) {
                    return conv_out(t.index(), v_offset + h.index() * HD + d).load();
                });
                auto decay = exp(g(t.index(), h.index()).load());
                auto b = beta(t.index(), h.index()).load();
                s = s * decay;
                // err[v] = v[v] - sum_k S[k,v] * k[k]
                auto sk = s * kv;// broadcast kv over vv
                auto retrieved = reduce(sk, kk, add, reduction::fold_left);// (vv)
                auto err = vvec - retrieved;
                // S += beta * outer(k, err)
                s = s + (kv * b) * err;// (kk) * (vv) outer via broadcast
                // o[v] = sum_k S[k,v] * q[k]
                auto o = reduce(s * qv, kk, add, reduction::fold_left);
                out(coord(t.index(), h.index(), 0), shape(one, one2, vv)).store(reshape(o, shape(one, one2, vv)));
            }
            state(coord(h.index(), 0, 0), shape(one, kk, vv)).store(reshape(s, shape(one, kk, vv)));
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, QK, HD), tensor_shape(T, QK, HD), tensor_shape(T, qkv_stride),
        tensor_shape(T, VH), tensor_shape(T, VH), tensor_shape(VH, HD, HD), tensor_shape(T, VH, HD)));
}

// Column-parallel delta-rule recurrence (executor integration extension).
// The [hd,hd] state tile SSA carry in capture_tile_gdn_recurrent exceeds the
// DX/VK XIR per-worker snapshot budget at the model head_dim=128 (the state
// alone is 64 KiB and the bridge double-buffers loop carries). This variant
// keeps the state in the caller's buffer and assigns one program per
// (v_head, v_column): each program owns a single state column S[:,v], streams
// k/q serially with SCALAR carries only (zero static tile snapshots), and
// reads/writes its column in place. The per-token math is identical:
//   S <- exp(g)*S; err = v - S^T k; S += beta * k err^T; o = S^T q.
// Same argument layout as capture_tile_gdn_recurrent. The state buffer is
// read+write (in-place per column; columns are disjoint across programs).
TileCapture capture_tile_gdn_recurrent_columns(uint32_t tokens, uint32_t v_heads,
                                               uint32_t qk_heads, uint32_t head_dim) {
    if (!extent_ok(tokens) || !extent_ok(v_heads) || !extent_ok(qk_heads) || !extent_ok(head_dim) ||
        v_heads % qk_heads != 0) {
        return capture_failure(TileOpStatus::invalid_argument, "gdn recurrent columns extents out of range");
    }
    const auto T = static_cast<int64_t>(tokens);
    const auto VH = static_cast<int64_t>(v_heads);
    const auto QK = static_cast<int64_t>(qk_heads);
    const auto HD = static_cast<int64_t>(head_dim);
    const auto v_offset = 2 * QK * HD;
    const auto qkv_stride = v_offset + VH * HD;
    if (!volume_ok(tokens, static_cast<uint64_t>(qkv_stride)) || !volume_ok(v_heads, head_dim, head_dim)) {
        return capture_failure(TileOpStatus::overflow, "gdn recurrent columns volume overflow");
    }
    auto definition = tile_kernel("qwen_tile_gdn_recurrent_cols", [=](TensorView<const float, 3> q,
                                                                      TensorView<const float, 3> k,
                                                                      TensorView<const float, 2> conv_out,
                                                                      TensorView<const float, 2> beta,
                                                                      TensorView<const float, 2> g,
                                                                      TensorView<float, 3> state,
                                                                      TensorView<float, 3> out) {
        auto kk = axis("kk", HD);
        auto vv = axis("vv", HD);
        auto h_axis = axis("h", VH);
        for (auto &prog : parallel(shape(h_axis, vv))) {
            auto h = prog.index(h_axis);
            auto v = prog.index(vv);
            auto qk_head = h % QK;// ggml tiled broadcast
            for (auto &t : serial(shape(T))) {
                auto decay = exp(g(t.index(), h).load());
                auto b = beta(t.index(), h).load();
                auto vval = conv_out(t.index(), v_offset + h * HD + v).load();
                // err = v_v - sum_k decay*S[k,v]*k[k]
                auto err = Scalar<float>{0.0f};
                for (auto &ki : serial(shape(kk))) {
                    auto skv = state(h, ki.index(), v).load() * decay;
                    auto kv = k(t.index(), qk_head, ki.index()).load();
                    err = err + skv * kv;
                }
                err = vval - err;
                // S[k,v] = decay*S[k,v] + beta*k[k]*err; o_v = sum_k S[k,v]*q[k]
                auto ov = Scalar<float>{0.0f};
                for (auto &ki : serial(shape(kk))) {
                    auto kv = k(t.index(), qk_head, ki.index()).load();
                    auto qv = q(t.index(), qk_head, ki.index()).load();
                    auto skv = state(h, ki.index(), v).load() * decay + b * kv * err;
                    state(h, ki.index(), v).store(skv);
                    ov = ov + skv * qv;
                }
                out(t.index(), h, v).store(ov);
            }
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, QK, HD), tensor_shape(T, QK, HD), tensor_shape(T, qkv_stride),
        tensor_shape(T, VH), tensor_shape(T, VH), tensor_shape(VH, HD, HD), tensor_shape(T, VH, HD)));
}

TileOpResult tile_ops_chunk_prefill_status() noexcept {
    return TileOpResult::error(TileOpStatus::unsupported,
                               "chunk-64 WY/UT GDN prefill is not implemented in the Tile operators; "
                               "capture_tile_gdn_recurrent with tokens>1 is the exact recurrent prefill path");
}

TileCapture capture_tile_gdn_gated_norm(uint32_t tokens, uint32_t heads, uint32_t head_dim, float eps) {
    if (!extent_ok(tokens) || !extent_ok(heads) || !extent_ok(head_dim) ||
        !volume_ok(tokens, heads, head_dim) || !(eps > 0.0f)) {
        return capture_failure(TileOpStatus::invalid_argument, "gdn gated norm extents out of range");
    }
    const auto T = static_cast<int64_t>(tokens);
    const auto H = static_cast<int64_t>(heads);
    const auto HD = static_cast<int64_t>(head_dim);
    const auto inv_hd = 1.0f / static_cast<float>(head_dim);
    auto definition = tile_kernel("qwen_tile_gdn_gated_norm", [=](TensorView<const float, 3> o,
                                                                  TensorView<const float, 2> z,
                                                                  TensorView<const float, 1> norm_w,
                                                                  TensorView<float, 3> out) {
        auto one = axis("one", 1);
        auto one2 = axis("one2", 1);
        auto d = axis("d", HD);
        auto t_axis = axis("t", T);
        auto h_axis = axis("h", H);
        for (auto &g : parallel(shape(t_axis, h_axis))) {
            auto t = g.index(t_axis);
            auto h = g.index(h_axis);
            auto ov = reshape(o.tile(coord(t, h, 0), shape(one, one2, d)).load(), shape(d));
            auto mean_square = reduce(ov * ov, d, add, reduction::fold_left).scalar() * inv_hd;
            auto scale = 1.0f / sqrt(mean_square + eps);
            auto wv = norm_w.tile(coord(0), shape(d)).load();
            auto zv = reshape(z.tile(coord(t, h * HD), shape(one, d)).load(), shape(d));
            auto normed = ov * scale * wv * tile_silu(zv);
            out(coord(t, h, 0), shape(one, one2, d)).store(reshape(normed, shape(one, one2, d)));
        }
    });
    return finish_capture(definition.capture(
        tensor_shape(T, H, HD), tensor_shape(T, H * HD), tensor_shape(HD), tensor_shape(T, H, HD)));
}



// ---------------------------------------------------------------------------
// 8. CPU oracles. Independent scalar code paths; double accumulation where
// the device reduction order is unspecified (mma/reduce trees), exact FP32
// step order where the device uses ordered serial carries.
// ---------------------------------------------------------------------------
namespace {

[[nodiscard]] double host_silu(double x) noexcept { return x / (1.0 + std::exp(-x)); }
[[nodiscard]] double host_sigmoid(double x) noexcept { return 1.0 / (1.0 + std::exp(-x)); }
[[nodiscard]] double host_softplus(double x, double threshold) noexcept {
    return x > threshold ? x : std::log1p(std::exp(-std::fabs(x))) + std::fmax(x, 0.0);
}

}// namespace

TileOpResult ref_tile_embedding(QwenTileWeightFormat format, luisa::span<const int32_t> ids,
                                luisa::span<const uint32_t> table, luisa::span<const float> scales,
                                uint32_t hidden, uint32_t vocab, luisa::vector<float> &out) {
    if (hidden == 0 || vocab == 0) { return invalid("zero embedding dimension"); }
    auto words = tile_packed_matrix_words(format, hidden, vocab);
    if (words == 0 || table.size() < words) { return invalid("embedding table too small"); }
    if (format == QwenTileWeightFormat::fp8_e4m3) {
        if (hidden % 4 != 0 || hidden % kTileFp8Group != 0) { return invalid("fp8 embedding requires hidden divisible by 64"); }
        if (scales.size() < tile_fp8_scale_elements(hidden, vocab)) { return invalid("scales too small"); }
    }
    out.assign(static_cast<size_t>(ids.size()) * hidden, 0.0f);
    for (size_t t = 0; t < ids.size(); t++) {
        auto id = ids[t];
        if (id < 0 || static_cast<uint64_t>(id) >= vocab) { continue; }// zero row
        for (uint32_t c = 0; c < hidden; c++) {
            out[t * hidden + c] = tile_host_decode_packed(format, table, scales, hidden,
                                                          static_cast<uint64_t>(id) * hidden + c);
        }
    }
    return TileOpResult::success();
}

TileOpResult ref_tile_rmsnorm(luisa::span<const float> x, luisa::span<const float> w,
                              luisa::span<const float> residual, bool has_residual,
                              uint32_t rows, uint32_t dim, float eps,
                              luisa::vector<float> &y, luisa::vector<float> &sum_out) {
    if (rows == 0 || dim == 0) { return invalid("zero rmsnorm dimension"); }
    size_t count = static_cast<size_t>(rows) * dim;
    if (x.size() < count || w.size() < dim) { return invalid("rmsnorm spans too small"); }
    if (has_residual && residual.size() < count) { return invalid("residual span too small"); }
    y.assign(count, 0.0f);
    sum_out.assign(has_residual ? count : 0, 0.0f);
    for (uint32_t r = 0; r < rows; r++) {
        double ss = 0.0;
        for (uint32_t c = 0; c < dim; c++) {
            double s = x[r * dim + c];
            if (has_residual) {
                s += residual[r * dim + c];
                sum_out[r * dim + c] = static_cast<float>(s);
            }
            ss += s * s;
        }
        auto scale = 1.0 / std::sqrt(ss / dim + eps);
        for (uint32_t c = 0; c < dim; c++) {
            double s = has_residual ? sum_out[r * dim + c] : x[r * dim + c];
            y[r * dim + c] = static_cast<float>(s * scale * w[c]);
        }
    }
    return TileOpResult::success();
}

TileOpResult ref_tile_matmul(QwenTileWeightFormat format, luisa::span<const float> x,
                             luisa::span<const uint32_t> w, luisa::span<const float> scales,
                             luisa::span<const float> residual, bool has_residual,
                             uint32_t m, uint32_t n, uint32_t k, luisa::vector<float> &y) {
    if (m == 0 || n == 0 || k == 0) { return invalid("zero matmul dimension"); }
    auto words = tile_packed_matrix_words(format, k, n);
    if (words == 0 || w.size() < words) { return invalid("weight words too small"); }
    if (x.size() < static_cast<size_t>(m) * k) { return invalid("x span too small"); }
    if (format == QwenTileWeightFormat::fp8_e4m3 &&
        scales.size() < tile_fp8_scale_elements(k, n)) { return invalid("scales too small"); }
    if (has_residual && residual.size() < static_cast<size_t>(m) * n) { return invalid("residual too small"); }
    y.assign(static_cast<size_t>(m) * n, 0.0f);
    for (uint32_t mi = 0; mi < m; mi++) {
        for (uint32_t ni = 0; ni < n; ni++) {
            double acc = has_residual ? residual[mi * n + ni] : 0.0;
            for (uint32_t ki = 0; ki < k; ki++) {
                acc += static_cast<double>(x[mi * k + ki]) *
                       tile_host_decode_packed(format, w, scales, k, static_cast<uint64_t>(ni) * k + ki);
            }
            y[mi * n + ni] = static_cast<float>(acc);
        }
    }
    return TileOpResult::success();
}

TileOpResult ref_tile_swiglu(luisa::span<const float> gate, luisa::span<const float> up,
                             luisa::vector<float> &out) {
    if (gate.size() != up.size()) { return invalid("swiglu size mismatch"); }
    out.resize(gate.size());
    for (size_t i = 0; i < gate.size(); i++) {
        out[i] = static_cast<float>(host_silu(gate[i]) * up[i]);
    }
    return TileOpResult::success();
}

TileOpResult ref_tile_swiglu_fused(luisa::span<const float> proj, uint32_t m, uint32_t f,
                                   luisa::vector<float> &out) {
    if (f == 0 || proj.size() < static_cast<size_t>(m) * 2 * f) { return invalid("swiglu fused span too small"); }
    out.assign(static_cast<size_t>(m) * f, 0.0f);
    for (uint32_t mi = 0; mi < m; mi++) {
        for (uint32_t fi = 0; fi < f; fi++) {
            out[mi * f + fi] = static_cast<float>(host_silu(proj[mi * 2 * f + fi]) *
                                                  proj[mi * 2 * f + f + fi]);
        }
    }
    return TileOpResult::success();
}

TileOpResult ref_tile_attn_prepare(luisa::span<const float> q_proj, luisa::span<const float> k_proj,
                                   luisa::span<const float> v_proj, luisa::span<const float> q_norm_w,
                                   luisa::span<const float> k_norm_w, uint32_t tokens,
                                   const QwenTileConfig &cfg,
                                   luisa::vector<float> &q, luisa::vector<float> &gate,
                                   luisa::vector<float> &k, luisa::vector<float> &v) {
    if (auto cv = validate_qwen_tile_config(cfg); !cv.ok()) { return cv; }
    const auto QH = cfg.q_heads, KH = cfg.kv_heads, HD = cfg.attn_head_dim;
    size_t qn = static_cast<size_t>(tokens) * QH * HD, kn = static_cast<size_t>(tokens) * KH * HD;
    if (q_proj.size() < static_cast<size_t>(tokens) * 2 * QH * HD ||
        k_proj.size() < kn || v_proj.size() < kn ||
        q_norm_w.size() < HD || k_norm_w.size() < HD) { return invalid("attn prepare spans too small"); }
    q.assign(qn, 0.0f);
    gate.assign(qn, 0.0f);
    k.assign(kn, 0.0f);
    v.assign(kn, 0.0f);
    auto norm_row = [&](const float *src, const float *w, float *dst) {
        double ss = 0.0;
        for (uint32_t c = 0; c < HD; c++) { ss += static_cast<double>(src[c]) * src[c]; }
        auto scale = 1.0 / std::sqrt(ss / HD + cfg.norm_eps);
        for (uint32_t c = 0; c < HD; c++) {
            dst[c] = static_cast<float>(src[c] * scale * w[c]);
        }
    };
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < QH; h++) {
            auto base = (static_cast<size_t>(t) * QH + h) * HD;
            auto src = q_proj.data() + (static_cast<size_t>(t) * 2 * QH + 2 * h) * HD;
            norm_row(src, q_norm_w.data(), q.data() + base);
            std::copy_n(src + HD, HD, gate.data() + base);
        }
        for (uint32_t h = 0; h < KH; h++) {
            auto base = (static_cast<size_t>(t) * KH + h) * HD;
            norm_row(k_proj.data() + base, k_norm_w.data(), k.data() + base);
            std::copy_n(v_proj.data() + base, HD, v.data() + base);
        }
    }
    return TileOpResult::success();
}

TileOpResult ref_tile_rope_partial(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                   uint32_t tokens, uint32_t heads, uint32_t head_dim,
                                   uint32_t rope_dim, float theta, luisa::vector<float> &out) {
    if (head_dim == 0 || rope_dim % 2 != 0 || rope_dim > head_dim) { return invalid("bad rope dims"); }
    if (positions.size() < tokens) { return invalid("positions too small"); }
    size_t count = static_cast<size_t>(tokens) * heads * head_dim;
    if (x.size() < count) { return invalid("rope x too small"); }
    luisa::vector<float> inv_freq;
    if (auto r = tile_host_rope_inv_freq(rope_dim, theta, inv_freq); !r.ok()) { return r; }
    out.assign(x.begin(), x.begin() + count);
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < heads; h++) {
            auto base = (static_cast<size_t>(t) * heads + h) * head_dim;
            // NeoX rotate-half: pairs (i, i + rope_dim/2); tail passthrough.
            const auto half = rope_dim / 2;
            for (uint32_t i = 0; i < half; i++) {
                float s, c;
                tile_host_rope_sin_cos(static_cast<float>(positions[t]) * inv_freq[i], s, c);
                auto x0 = x[base + i], x1 = x[base + i + half];
                out[base + i] = x0 * c - x1 * s;
                out[base + i + half] = x0 * s + x1 * c;
            }
        }
    }
    return TileOpResult::success();
}

TileOpResult ref_tile_kv_append(luisa::span<const float> k, luisa::span<const float> v,
                                uint32_t tokens, uint32_t channels, uint32_t pos0,
                                luisa::span<float> k_cache, luisa::span<float> v_cache) {
    size_t count = static_cast<size_t>(tokens) * channels;
    if (k.size() < count || v.size() < count) { return invalid("kv append src too small"); }
    size_t end = (static_cast<size_t>(pos0) + tokens) * channels;
    if (k_cache.size() < end || v_cache.size() < end) { return invalid("kv cache too small"); }
    for (uint32_t t = 0; t < tokens; t++) {
        std::copy_n(k.data() + static_cast<size_t>(t) * channels, channels,
                    k_cache.data() + (static_cast<size_t>(pos0) + t) * channels);
        std::copy_n(v.data() + static_cast<size_t>(t) * channels, channels,
                    v_cache.data() + (static_cast<size_t>(pos0) + t) * channels);
    }
    return TileOpResult::success();
}

namespace {

// Two-pass softmax reference (independent of the device online algorithm).
[[nodiscard]] TileOpResult attn_reference(luisa::span<const float> q, luisa::span<const float> gate,
                                          luisa::span<const float> k_cache, luisa::span<const float> v_cache,
                                          uint32_t tokens, uint32_t q_heads, uint32_t kv_heads,
                                          uint32_t head_dim, float scale, uint32_t pos0,
                                          uint32_t kv_len, luisa::vector<float> &out) {
    if (q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 || head_dim == 0) { return invalid("bad attn dims"); }
    if (kv_len == 0 || static_cast<uint64_t>(pos0) + tokens > kv_len) { return invalid("bad causal window"); }
    size_t qn = static_cast<size_t>(tokens) * q_heads * head_dim;
    size_t cn = static_cast<size_t>(kv_len) * kv_heads * head_dim;
    if (q.size() < qn || gate.size() < qn || k_cache.size() < cn || v_cache.size() < cn) {
        return invalid("attn spans too small");
    }
    auto group = q_heads / kv_heads;
    out.assign(qn, 0.0f);
    luisa::vector<double> scores(kv_len);
    for (uint32_t t = 0; t < tokens; t++) {
        auto limit = pos0 + t + 1;
        for (uint32_t h = 0; h < q_heads; h++) {
            auto kvh = h / group;
            auto qbase = (static_cast<size_t>(t) * q_heads + h) * head_dim;
            double peak = -std::numeric_limits<double>::infinity();
            for (uint32_t j = 0; j < limit; j++) {
                double dot = 0.0;
                auto kbase = (static_cast<size_t>(j) * kv_heads + kvh) * head_dim;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += static_cast<double>(q[qbase + d]) * k_cache[kbase + d];
                }
                scores[j] = dot * scale;
                peak = std::fmax(peak, scores[j]);
            }
            double sum = 0.0;
            for (uint32_t j = 0; j < limit; j++) {
                scores[j] = std::exp(scores[j] - peak);
                sum += scores[j];
            }
            for (uint32_t d = 0; d < head_dim; d++) {
                double acc = 0.0;
                for (uint32_t j = 0; j < limit; j++) {
                    acc += scores[j] * v_cache[(static_cast<size_t>(j) * kv_heads + kvh) * head_dim + d];
                }
                out[qbase + d] = static_cast<float>(acc / sum * host_sigmoid(gate[qbase + d]));
            }
        }
    }
    return TileOpResult::success();
}

}// namespace

TileOpResult ref_tile_attn_prefill(luisa::span<const float> q, luisa::span<const float> gate,
                                   luisa::span<const float> k_cache, luisa::span<const float> v_cache,
                                   uint32_t tokens, uint32_t q_heads, uint32_t kv_heads,
                                   uint32_t head_dim, float scale, uint32_t pos0,
                                   uint32_t kv_len, luisa::vector<float> &out) {
    return attn_reference(q, gate, k_cache, v_cache, tokens, q_heads, kv_heads, head_dim, scale, pos0, kv_len, out);
}

TileOpResult ref_tile_attn_decode(luisa::span<const float> q, luisa::span<const float> gate,
                                  luisa::span<const float> k_cache, luisa::span<const float> v_cache,
                                  uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                                  float scale, uint32_t kv_len, luisa::vector<float> &out) {
    return attn_reference(q, gate, k_cache, v_cache, 1, q_heads, kv_heads, head_dim, scale, kv_len - 1, kv_len, out);
}

TileOpResult ref_tile_gdn_conv(luisa::span<const float> src, luisa::span<const float> weight,
                               luisa::span<float> ring, uint32_t tokens, uint32_t channels,
                               uint32_t kernel_size, uint32_t pos0, luisa::vector<float> &out) {
    if (kernel_size < 2 || channels == 0) { return invalid("bad conv dims"); }
    size_t count = static_cast<size_t>(tokens) * channels;
    auto R = kernel_size - 1;
    if (src.size() < count || weight.size() < static_cast<size_t>(kernel_size) * channels ||
        ring.size() < static_cast<size_t>(channels) * R) { return invalid("conv spans too small"); }
    out.assign(count, 0.0f);
    for (uint32_t t = 0; t < tokens; t++) {
        auto pos = static_cast<int64_t>(pos0) + t;
        for (uint32_t c = 0; c < channels; c++) {
            double acc = 0.0;
            for (uint32_t j = 0; j < kernel_size; j++) {
                auto src_pos = pos - static_cast<int64_t>(R) + j;
                double value = 0.0;
                if (src_pos >= static_cast<int64_t>(pos0)) {
                    value = src[static_cast<size_t>(src_pos - pos0) * channels + c];
                } else if (src_pos >= 0) {
                    value = ring[static_cast<size_t>(c) * R + static_cast<uint64_t>(src_pos) % R];
                }
                // GGUF ssm_conv1d [kernel, channels] payload is CHANNEL-major:
                  // weight[c*kernel + j] is tap j of channel c.
                  acc += value * weight[static_cast<size_t>(c) * kernel_size + j];
            }
            out[static_cast<size_t>(t) * channels + c] = static_cast<float>(host_silu(acc));
        }
        // push pre-activation values into the ring AFTER the output.
        for (uint32_t c = 0; c < channels; c++) {
            ring[static_cast<size_t>(c) * R + static_cast<uint64_t>(pos) % R] =
                src[static_cast<size_t>(t) * channels + c];
        }
    }
    return TileOpResult::success();
}

TileOpResult ref_tile_gdn_gates(luisa::span<const float> a, luisa::span<const float> b,
                                luisa::span<const float> ssm_a, luisa::span<const float> dt_bias,
                                uint32_t tokens, uint32_t heads, float softplus_threshold,
                                luisa::vector<float> &beta, luisa::vector<float> &g) {
    size_t count = static_cast<size_t>(tokens) * heads;
    if (heads == 0 || a.size() < count || b.size() < count ||
        ssm_a.size() < heads || dt_bias.size() < heads) { return invalid("gates spans too small"); }
    beta.assign(count, 0.0f);
    g.assign(count, 0.0f);
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < heads; h++) {
            auto i = static_cast<size_t>(t) * heads + h;
            beta[i] = static_cast<float>(host_sigmoid(b[i]));
            // GGUF ssm_a is already -exp(A_log): direct multiply.
            g[i] = static_cast<float>(static_cast<double>(ssm_a[h]) *
                                      host_softplus(a[i] + dt_bias[h], softplus_threshold));
        }
    }
    return TileOpResult::success();
}

TileOpResult ref_tile_gdn_qk_norm(luisa::span<const float> conv_out, uint32_t tokens,
                                  uint32_t qk_heads, uint32_t head_dim, uint32_t v_channels,
                                  float eps, float q_scale,
                                  luisa::vector<float> &q, luisa::vector<float> &k) {
    if (qk_heads == 0 || head_dim == 0) { return invalid("bad qk norm dims"); }
    auto stride = static_cast<size_t>(2 * qk_heads * head_dim + v_channels);
    if (conv_out.size() < static_cast<size_t>(tokens) * stride) { return invalid("conv_out too small"); }
    size_t count = static_cast<size_t>(tokens) * qk_heads * head_dim;
    q.assign(count, 0.0f);
    k.assign(count, 0.0f);
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < qk_heads; h++) {
            for (uint32_t s = 0; s < 2; s++) {
                auto src = conv_out.data() + t * stride + s * qk_heads * head_dim + h * head_dim;
                double ss = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) { ss += static_cast<double>(src[d]) * src[d]; }
                auto inv = 1.0 / std::sqrt(ss + static_cast<double>(eps));
                auto dst = (s == 0 ? q.data() : k.data()) + (static_cast<size_t>(t) * qk_heads + h) * head_dim;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dst[d] = static_cast<float>(src[d] * inv * (s == 0 ? q_scale : 1.0f));
                }
            }
        }
    }
    return TileOpResult::success();
}

TileOpResult ref_tile_gdn_recurrent(luisa::span<const float> q, luisa::span<const float> k,
                                    luisa::span<const float> conv_out, luisa::span<const float> beta,
                                    luisa::span<const float> g, luisa::span<float> state,
                                    uint32_t tokens, uint32_t v_heads, uint32_t head_dim,
                                    uint32_t qk_heads, luisa::vector<float> &out) {
    if (v_heads == 0 || qk_heads == 0 || head_dim == 0 || v_heads % qk_heads != 0) { return invalid("bad recurrent dims"); }
    auto HD = static_cast<size_t>(head_dim);
    auto v_offset = static_cast<size_t>(2 * qk_heads * head_dim);
    auto qkv_stride = v_offset + static_cast<size_t>(v_heads) * head_dim;
    size_t qn = static_cast<size_t>(tokens) * qk_heads * head_dim;
    if (q.size() < qn || k.size() < qn ||
        conv_out.size() < static_cast<size_t>(tokens) * qkv_stride ||
        beta.size() < static_cast<size_t>(tokens) * v_heads ||
        g.size() < static_cast<size_t>(tokens) * v_heads ||
        state.size() < static_cast<size_t>(v_heads) * HD * HD) { return invalid("recurrent spans too small"); }
    out.assign(static_cast<size_t>(tokens) * v_heads * head_dim, 0.0f);
    luisa::vector<double> s(HD * HD), err(HD), o(HD);
    for (uint32_t h = 0; h < v_heads; h++) {
        auto qk_head = h % qk_heads;// ggml tiled broadcast
        auto sbase = static_cast<size_t>(h) * HD * HD;
        for (size_t i = 0; i < HD * HD; i++) { s[i] = state[sbase + i]; }
        for (uint32_t t = 0; t < tokens; t++) {
            auto qbase = (static_cast<size_t>(t) * qk_heads + qk_head) * HD;
            auto vbase = static_cast<size_t>(t) * qkv_stride + v_offset + static_cast<size_t>(h) * HD;
            auto decay = std::exp(static_cast<double>(g[static_cast<size_t>(t) * v_heads + h]));
            auto b = static_cast<double>(beta[static_cast<size_t>(t) * v_heads + h]);
            for (size_t i = 0; i < HD * HD; i++) { s[i] *= decay; }
            for (size_t v = 0; v < HD; v++) {
                double retrieved = 0.0;
                for (size_t kk = 0; kk < HD; kk++) { retrieved += s[kk * HD + v] * k[qbase + kk]; }
                err[v] = conv_out[vbase + v] - retrieved;
            }
            for (size_t kk = 0; kk < HD; kk++) {
                auto bk = b * k[qbase + kk];
                for (size_t v = 0; v < HD; v++) { s[kk * HD + v] += bk * err[v]; }
            }
            for (size_t v = 0; v < HD; v++) {
                double acc = 0.0;
                for (size_t kk = 0; kk < HD; kk++) { acc += s[kk * HD + v] * q[qbase + kk]; }
                out[(static_cast<size_t>(t) * v_heads + h) * HD + v] = static_cast<float>(acc);
            }
        }
        for (size_t i = 0; i < HD * HD; i++) { state[sbase + i] = static_cast<float>(s[i]); }
    }
    return TileOpResult::success();
}

TileOpResult ref_tile_gdn_gated_norm(luisa::span<const float> o, luisa::span<const float> z,
                                     luisa::span<const float> norm_w, uint32_t tokens,
                                     uint32_t heads, uint32_t head_dim, float eps,
                                     luisa::vector<float> &out) {
    if (heads == 0 || head_dim == 0) { return invalid("bad gated norm dims"); }
    size_t count = static_cast<size_t>(tokens) * heads * head_dim;
    if (o.size() < count || z.size() < count || norm_w.size() < head_dim) { return invalid("gated norm spans too small"); }
    out.assign(count, 0.0f);
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < heads; h++) {
            auto base = (static_cast<size_t>(t) * heads + h) * head_dim;
            double ss = 0.0;
            for (uint32_t d = 0; d < head_dim; d++) { ss += static_cast<double>(o[base + d]) * o[base + d]; }
            auto scale = 1.0 / std::sqrt(ss / head_dim + eps);
            for (uint32_t d = 0; d < head_dim; d++) {
                out[base + d] = static_cast<float>(o[base + d] * scale * norm_w[d] * host_silu(z[base + d]));
            }
        }
    }
    return TileOpResult::success();
}

// ---------------------------------------------------------------------------
// 9. Device-free self-test.
// ---------------------------------------------------------------------------
namespace {

struct SelfTest {
    luisa::vector<luisa::string> *report;
    bool all_ok{true};

    void check(bool ok, luisa::string name) {
        all_ok &= ok;
        if (report) { report->emplace_back(luisa::string(ok ? "[ok] " : "[FAIL] ") + name); }
    }
    void check_capture(TileCapture &c, const char *name) {
        auto ok = c.ok();
        all_ok &= ok;
        if (report) {
            report->emplace_back(luisa::string(ok ? "[ok] " : "[FAIL] ") + "capture " + name +
                                 (ok ? "" : (": " + c.status.message)));
            if (!ok) {
                for (auto &d : c.diagnostics) { report->emplace_back("    diag: " + d); }
            }
        }
    }
};

[[nodiscard]] bool close_float(float a, float b, float atol, float rtol) noexcept {
    if (!std::isfinite(a) || !std::isfinite(b)) { return false; }
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}
[[nodiscard]] bool close_span(luisa::span<const float> a, luisa::span<const float> b,
                              float atol, float rtol) noexcept {
    if (a.size() != b.size()) { return false; }
    for (size_t i = 0; i < a.size(); i++) {
        if (!close_float(a[i], b[i], atol, rtol)) { return false; }
    }
    return true;
}

}// namespace

TileOpResult tile_ops_self_test(luisa::vector<luisa::string> *report) {
    SelfTest st{report};

    // -- codec round-trips ---------------------------------------------------
    {
        bool ok = true;
        for (auto v : {0.0f, -0.0f, 1.0f, -1.0f, 0.1f, 3.14159f, 65504.0f, 1e-8f, -2.5e-7f}) {
            auto bits = tile_host_f32_to_bf16_bits(v);
            auto back = tile_host_bf16_bits_to_f32(bits);
            ok &= std::fabs(back - v) <= 0.004f * std::fabs(v) + 1e-40f;
        }
        ok &= tile_host_bf16_bits_to_f32(0x3f80u) == 1.0f;
        ok &= tile_host_f32_to_bf16_bits(1.0f) == 0x3f80u;
        st.check(ok, "bf16 codec round-trip");
    }
    {
        bool ok = true;
        // every finite E4M3 code decodes to a finite value; encode(decode) is
        // the identity for all finite codes.
        for (uint32_t code = 0; code < 256; code++) {
            if ((code & 0x7fu) == 0x7fu) {
                ok &= std::isnan(tile_host_fp8_e4m3_bits_to_f32(static_cast<uint8_t>(code)));
                continue;
            }
            auto v = tile_host_fp8_e4m3_bits_to_f32(static_cast<uint8_t>(code));
            ok &= std::isfinite(v);
            ok &= tile_host_f32_to_fp8_e4m3_bits(v) == static_cast<uint8_t>(code);
        }
        ok &= tile_host_fp8_e4m3_bits_to_f32(0x7eu) == 448.0f;
        ok &= tile_host_f32_to_fp8_e4m3_bits(1e30f) == 0x7eu;// saturate
        ok &= tile_host_f32_to_fp8_e4m3_bits(-1e30f) == 0xfeu;
        ok &= tile_host_f32_to_fp8_e4m3_bits(0.0f) == 0x00u;
        st.check(ok, "fp8 e4m3 codec exhaustive finite round-trip");
    }

    // -- packed matrix round-trip + oracle GEMM (ragged K/N, partial groups) --
    for (auto format : {QwenTileWeightFormat::bf16, QwenTileWeightFormat::fp8_e4m3}) {
        const uint32_t K = 130, N = 7, M = 3;// ragged: K%64=2, N odd
        luisa::vector<float> w(static_cast<size_t>(N) * K), x(static_cast<size_t>(M) * K);
        for (size_t i = 0; i < w.size(); i++) { w[i] = std::sin(static_cast<float>(i) * 0.13f) * 0.25f; }
        for (size_t i = 0; i < x.size(); i++) { x[i] = std::cos(static_cast<float>(i) * 0.31f) * 0.5f; }
        // zero group: force one full group to zero
        for (uint32_t kk = 64; kk < 128; kk++) { w[2 * K + kk] = 0.0f; }
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        auto pack = tile_host_pack_matrix(format, w, K, N, words, scales);
        st.check(pack.ok(), luisa::string("pack matrix ") + qwen_tile_weight_format_name(format).data());
        // decode parity for every element
        bool decode_ok = true;
        double max_err = 0.0;
        for (size_t i = 0; i < w.size(); i++) {
            auto decoded = tile_host_decode_packed(format, words, scales, K, i);
            auto err = std::fabs(decoded - w[i]);
            max_err = std::fmax(max_err, err);
            if (format == QwenTileWeightFormat::bf16) { decode_ok &= err <= 0.004 * std::fabs(w[i]); }
        }
        if (format == QwenTileWeightFormat::fp8_e4m3) {
            // E4M3 relative step is 2^-3; per-group scale = max_abs/448 gives
            // a worst-case absolute error of ~max_abs * 2^-4 / 448 * 448/2...
            // empirically bounded by 6.5% of the group max for this data.
            decode_ok = max_err <= 0.02;
        }
        st.check(decode_ok, luisa::string("packed decode parity ") + qwen_tile_weight_format_name(format).data());
        luisa::vector<float> y;
        auto mm = ref_tile_matmul(format, x, words, scales, {}, false, M, N, K, y);
        bool mm_ok = mm.ok();
        for (uint32_t mi = 0; mi < M && mm_ok; mi++) {
            for (uint32_t ni = 0; ni < N && mm_ok; ni++) {
                double want = 0.0;
                for (uint32_t ki = 0; ki < K; ki++) {
                    want += static_cast<double>(x[mi * K + ki]) *
                            tile_host_decode_packed(format, words, scales, K, static_cast<size_t>(ni) * K + ki);
                }
                mm_ok &= close_float(y[mi * N + ni], static_cast<float>(want), 1e-4f, 1e-4f);
            }
        }
        st.check(mm_ok, luisa::string("oracle matmul ") + qwen_tile_weight_format_name(format).data());
        // embedding oracle with an out-of-range id (hidden=128: whole fp8
        // words and scale groups per row, as the capture requires).
        const uint32_t EH = 128;
        luisa::vector<float> ew(static_cast<size_t>(N) * EH);
        for (size_t i = 0; i < ew.size(); i++) { ew[i] = std::sin(static_cast<float>(i) * 0.17f) * 0.25f; }
        luisa::vector<uint32_t> ewords;
        luisa::vector<float> escales;
        auto ep = tile_host_pack_matrix(format, ew, EH, N, ewords, escales);
        luisa::vector<int32_t> ids{0, 6, 99, 3};
        luisa::vector<float> emb;
        auto er = ref_tile_embedding(format, ids, ewords, escales, EH, N, emb);
        bool emb_ok = ep.ok() && er.ok() && emb.size() == ids.size() * EH;
        for (size_t c = 0; c < EH && emb_ok; c++) { emb_ok &= emb[2 * EH + c] == 0.0f; }
        for (size_t c = 0; c < EH && emb_ok; c++) {
            emb_ok &= close_float(emb[c], tile_host_decode_packed(format, ewords, escales, EH, c), 1e-6f, 1e-6f);
            emb_ok &= close_float(emb[EH + c], tile_host_decode_packed(format, ewords, escales, EH, 6 * EH + c), 1e-6f, 1e-6f);
        }
        st.check(emb_ok, luisa::string("oracle embedding ") + qwen_tile_weight_format_name(format).data());
    }

    // -- rmsnorm (both epilogues) --------------------------------------------
    {
        const uint32_t R = 3, D = 33;
        luisa::vector<float> x(R * D), w(D), res(R * D);
        for (size_t i = 0; i < x.size(); i++) { x[i] = std::sin(i * 0.7f); }
        for (size_t i = 0; i < w.size(); i++) { w[i] = std::cos(i * 0.3f) * 0.1f; }
        for (size_t i = 0; i < res.size(); i++) { res[i] = std::sin(i * 0.11f) * 0.5f; }
        luisa::vector<float> y, sum;
        auto r1 = ref_tile_rmsnorm(x, w, {}, false, R, D, 1e-6f, y, sum);
        bool ok = r1.ok() && sum.empty();
        // GGUF pre-added convention: w=1 must give plain RMS normalization
        // (the converter folds HF's 1+w offline; w=0 now means ZERO output).
        luisa::vector<float> w0(D, 1.0f), y0, s0;
        auto r2 = ref_tile_rmsnorm(x, w0, {}, false, R, D, 1e-6f, y0, s0);
        ok &= r2.ok();
        for (uint32_t r = 0; r < R && ok; r++) {
            double ss = 0.0;
            for (uint32_t c = 0; c < D; c++) { ss += static_cast<double>(x[r * D + c]) * x[r * D + c]; }
            auto scale = 1.0 / std::sqrt(ss / D + 1e-6);
            for (uint32_t c = 0; c < D; c++) {
                ok &= close_float(y0[r * D + c], static_cast<float>(x[r * D + c] * scale), 1e-5f, 1e-5f);
            }
        }
        auto r3 = ref_tile_rmsnorm(x, w, res, true, R, D, 1e-6f, y, sum);
        ok &= r3.ok() && sum.size() == R * D;
        for (size_t i = 0; i < sum.size() && ok; i++) { ok &= close_float(sum[i], x[i] + res[i], 1e-6f, 1e-6f); }
        st.check(ok, "oracle rmsnorm plain+residual, pre-added direct convention");
    }

    // -- swiglu ---------------------------------------------------------------
    {
        luisa::vector<float> g(77), u(77), out, fused(3 * 2 * 19), fout;
        for (size_t i = 0; i < g.size(); i++) { g[i] = std::sin(i * 0.2f) * 3.0f; u[i] = std::cos(i * 0.4f); }
        auto r = ref_tile_swiglu(g, u, out);
        bool ok = r.ok();
        for (size_t i = 0; i < g.size() && ok; i++) { ok &= close_float(out[i], g[i] / (1.0f + std::exp(-g[i])) * u[i], 1e-5f, 1e-5f); }
        for (size_t i = 0; i < fused.size(); i++) { fused[i] = std::sin(i * 0.17f) * 2.0f; }
        auto r2 = ref_tile_swiglu_fused(fused, 3, 19, fout);
        ok &= r2.ok();
        for (uint32_t mi = 0; mi < 3 && ok; mi++) {
            for (uint32_t fi = 0; fi < 19 && ok; fi++) {
                auto gv = fused[mi * 38 + fi], uv = fused[mi * 38 + 19 + fi];
                ok &= close_float(fout[mi * 19 + fi], gv / (1.0f + std::exp(-gv)) * uv, 1e-5f, 1e-5f);
            }
        }
        st.check(ok, "oracle swiglu split+fused");
    }

    // -- rope: shared sin/cos polynomial vs std::sin/cos ----------------------
    {
        bool ok = true;
        for (auto angle : {0.0f, 0.1f, 1.0f, 3.0f, 10.0f, 100.0f, 1000.0f, 12345.0f, -7.3f, 262143.0f}) {
            float s, c;
            tile_host_rope_sin_cos(angle, s, c);
            // FP32 Cody-Waite limit: ~2e-4 up to 1e4, ~4e-3 at 262144.
            auto tol = std::fabs(angle) > 1e5f ? 5e-3f : 2e-4f;
            ok &= std::fabs(s - std::sin(angle)) <= tol;
            ok &= std::fabs(c - std::cos(angle)) <= tol;
        }
        st.check(ok, "rope sin/cos polynomial vs libm");
        QwenTileConfig cfg;
        luisa::vector<float> x(2 * 3 * cfg.attn_head_dim), out;
        luisa::vector<int32_t> pos{5, 1000};
        for (size_t i = 0; i < x.size(); i++) { x[i] = std::sin(i * 0.23f); }
        auto r = ref_tile_rope_partial(x, pos, 2, 3, cfg.attn_head_dim, cfg.rope_dim, cfg.rope_theta, out);
        ok = r.ok();
        // tail passthrough
        for (uint32_t t = 0; t < 2 && ok; t++) {
            for (uint32_t h = 0; h < 3 && ok; h++) {
                auto base = (t * 3 + h) * cfg.attn_head_dim;
                for (uint32_t d = cfg.rope_dim; d < cfg.attn_head_dim; d++) { ok &= out[base + d] == x[base + d]; }
                // NeoX pair norm preserved by rotation: pairs (i, i+half)
                const auto half = cfg.rope_dim / 2;
                for (uint32_t i = 0; i < half; i++) {
                    auto n0 = x[base + i] * x[base + i] + x[base + i + half] * x[base + i + half];
                    auto n1 = out[base + i] * out[base + i] + out[base + i + half] * out[base + i + half];
                    ok &= close_float(n1, n0, 1e-4f, 1e-4f);
                }
            }
        }
        st.check(ok, "oracle partial rope tail+pair-norm");
    }

    // -- attention prepare + prefill/decode oracle ----------------------------
    {
        QwenTileConfig cfg;
        cfg.q_heads = 4; cfg.kv_heads = 2; cfg.attn_head_dim = 8; cfg.rope_dim = 4;
        const uint32_t T = 3, P0 = 2, CAP = 9, KV = 5;
        auto QN = cfg.attn_q_proj_n(), KVN = cfg.attn_kv_n(), HD = cfg.attn_head_dim;
        luisa::vector<float> qp(T * QN), kp(T * KVN), vp(T * KVN), qw(HD), kw(HD);
        for (size_t i = 0; i < qp.size(); i++) { qp[i] = std::sin(i * 0.19f); }
        for (size_t i = 0; i < kp.size(); i++) { kp[i] = std::cos(i * 0.23f); }
        for (size_t i = 0; i < vp.size(); i++) { vp[i] = std::sin(i * 0.31f + 1.0f); }
        for (size_t i = 0; i < qw.size(); i++) { qw[i] = 1.0f + 0.05f * std::sin(i * 0.7f); kw[i] = 1.0f + 0.05f * std::cos(i * 0.9f); }// pre-added GGUF weights
        luisa::vector<float> q, gate, k, v;
        auto r = ref_tile_attn_prepare(qp, kp, vp, qw, kw, T, cfg, q, gate, k, v);
        st.check(r.ok(), "oracle attn prepare");
        luisa::vector<float> kc(CAP * KVN, 0.0f), vc(CAP * KVN, 0.0f);
        auto ra = ref_tile_kv_append(k, v, T, KVN, P0, kc, vc);
        st.check(ra.ok(), "oracle kv append");
        luisa::vector<float> out;
        auto rp = ref_tile_attn_prefill(q, gate, kc, vc, T, cfg.q_heads, cfg.kv_heads, HD,
                                        cfg.attn_scale(), P0, KV, out);
        bool ok = rp.ok();
        // independent single-query decode at the last position must match
        luisa::vector<float> q1(q.end() - cfg.q_heads * HD, q.end());
        luisa::vector<float> g1(gate.end() - cfg.q_heads * HD, gate.end());
        luisa::vector<float> dout;
        auto rd = ref_tile_attn_decode(q1, g1, kc, vc, cfg.q_heads, cfg.kv_heads, HD,
                                       cfg.attn_scale(), KV, dout);
        ok &= rd.ok();
        for (size_t i = 0; i < dout.size() && ok; i++) {
            ok &= close_float(dout[i], out[out.size() - dout.size() + i], 2e-5f, 2e-5f);
        }
        st.check(ok, "oracle prefill last row == decode");
    }

    // -- GDN conv across chunk boundary + gates + qk norm ---------------------
    {
        const uint32_t C = 6, KS = 4, T1 = 3, T2 = 2;
        luisa::vector<float> src((T1 + T2) * C), w(KS * C), ring(C * (KS - 1), 0.0f);
        for (size_t i = 0; i < src.size(); i++) { src[i] = std::sin(i * 0.41f); }
        for (size_t i = 0; i < w.size(); i++) { w[i] = std::cos(i * 0.13f) * 0.3f; }
        luisa::vector<float> out1, out2, whole;
        auto r1 = ref_tile_gdn_conv(luisa::span{src}.first(T1 * C), w, ring, T1, C, KS, 0, out1);
        auto r2 = ref_tile_gdn_conv(luisa::span{src}.subspan(T1 * C), w, ring, T2, C, KS, T1, out2);
        luisa::vector<float> ring0(C * (KS - 1), 0.0f);
        auto r3 = ref_tile_gdn_conv(src, w, ring0, T1 + T2, C, KS, 0, whole);
        bool ok = r1.ok() && r2.ok() && r3.ok();
        for (size_t i = 0; i < out1.size() && ok; i++) { ok &= close_float(out1[i], whole[i], 1e-5f, 1e-5f); }
        for (size_t i = 0; i < out2.size() && ok; i++) { ok &= close_float(out2[i], whole[T1 * C + i], 1e-5f, 1e-5f); }
        st.check(ok, "oracle gdn conv chunked == whole (ring history)");

        const uint32_t H = 4;
        luisa::vector<float> a(T1 * H), b(T1 * H), ssm_a(H), dt(H), beta, g;
        for (size_t i = 0; i < a.size(); i++) { a[i] = std::sin(i * 0.5f) * 4.0f; b[i] = std::cos(i * 0.3f) * 2.0f; }
        for (uint32_t h = 0; h < H; h++) { ssm_a[h] = -0.5f * h; dt[h] = 0.1f * h; }
        auto rg = ref_tile_gdn_gates(a, b, ssm_a, dt, T1, H, 20.0f, beta, g);
        ok = rg.ok();
        for (size_t i = 0; i < g.size() && ok; i++) { ok &= g[i] <= 0.0f && beta[i] > 0.0f && beta[i] < 1.0f; }
        st.check(ok, "oracle gdn gates (g<=0, beta in (0,1))");

        const uint32_t QK = 2, HD = 8, VC = 4 * HD;
        luisa::vector<float> conv(T1 * (2 * QK * HD + VC)), q, k;
        for (size_t i = 0; i < conv.size(); i++) { conv[i] = std::sin(i * 0.17f); }
        auto rq = ref_tile_gdn_qk_norm(conv, T1, QK, HD, VC, 1e-6f, 1.0f / std::sqrt(8.0f), q, k);
        ok = rq.ok();
        for (uint32_t t = 0; t < T1 && ok; t++) {
            for (uint32_t h = 0; h < QK && ok; h++) {
                double qn = 0.0, kn = 0.0;
                for (uint32_t d = 0; d < HD; d++) {
                    qn += q[(t * QK + h) * HD + d] * q[(t * QK + h) * HD + d];
                    kn += k[(t * QK + h) * HD + d] * k[(t * QK + h) * HD + d];
                }
                ok &= close_float(static_cast<float>(qn), 1.0f / 8.0f, 1e-5f, 1e-5f);
                ok &= close_float(static_cast<float>(kn), 1.0f, 1e-5f, 1e-5f);
            }
        }
        st.check(ok, "oracle gdn qk l2 norm (+q scale)");
    }

    // -- GDN recurrence: chunked == whole; delta-rule sanity ------------------
    {
        const uint32_t VH = 4, QK = 2, HD = 4, T1 = 3, T2 = 2;
        auto stride = 2 * QK * HD + VH * HD;
        luisa::vector<float> q((T1 + T2) * QK * HD), k((T1 + T2) * QK * HD),
            conv((T1 + T2) * stride), beta((T1 + T2) * VH), g((T1 + T2) * VH);
        for (size_t i = 0; i < q.size(); i++) { q[i] = std::sin(i * 0.37f) * 0.3f; k[i] = std::cos(i * 0.29f) * 0.3f; }
        for (size_t i = 0; i < conv.size(); i++) { conv[i] = std::sin(i * 0.13f + 2.0f) * 0.5f; }
        for (size_t i = 0; i < beta.size(); i++) { beta[i] = 0.3f + 0.1f * std::sin(i * 0.7f); }
        for (size_t i = 0; i < g.size(); i++) { g[i] = -0.05f - 0.02f * std::cos(i * 0.3f); }
        luisa::vector<float> state(VH * HD * HD), s1, s2;
        for (size_t i = 0; i < state.size(); i++) { state[i] = std::sin(i * 0.11f) * 0.05f; }
        s1 = state; s2 = state;
        luisa::vector<float> o1, o2, ow;
        auto r1 = ref_tile_gdn_recurrent(luisa::span{q}.first(T1 * QK * HD), luisa::span{k}.first(T1 * QK * HD),
                                         luisa::span{conv}.first(T1 * stride), luisa::span{beta}.first(T1 * VH),
                                         luisa::span{g}.first(T1 * VH), s1, T1, VH, HD, QK, o1);
        auto r2 = ref_tile_gdn_recurrent(luisa::span{q}.subspan(T1 * QK * HD), luisa::span{k}.subspan(T1 * QK * HD),
                                         luisa::span{conv}.subspan(T1 * stride), luisa::span{beta}.subspan(T1 * VH),
                                         luisa::span{g}.subspan(T1 * VH), s1, T2, VH, HD, QK, o2);
        auto r3 = ref_tile_gdn_recurrent(q, k, conv, beta, g, s2, T1 + T2, VH, HD, QK, ow);
        bool ok = r1.ok() && r2.ok() && r3.ok();
        for (size_t i = 0; i < o1.size() && ok; i++) { ok &= close_float(o1[i], ow[i], 1e-5f, 1e-5f); }
        for (size_t i = 0; i < o2.size() && ok; i++) { ok &= close_float(o2[i], ow[o1.size() + i], 1e-5f, 1e-5f); }
        ok &= close_span(s1, s2, 1e-5f, 1e-5f);
        st.check(ok, "oracle gdn recurrence chunked == whole (state+out)");

        // delta-rule sanity: with beta=1, g=0, k unit basis e0, v arbitrary,
        // after one step S^T k must equal v exactly (error cancelled).
        luisa::vector<float> q1(QK * HD, 0.0f), k1(QK * HD, 0.0f), conv1(stride, 0.0f),
            b1(VH, 1.0f), g1(VH, 0.0f), s0(VH * HD * HD, 0.0f), o;
        k1[0] = 1.0f;// QK head 0, dim 0
        for (uint32_t d = 0; d < HD; d++) { conv1[2 * QK * HD + d] = 0.5f + 0.1f * d; }// V head 0
        auto rr = ref_tile_gdn_recurrent(q1, k1, conv1, b1, g1, s0, 1, VH, HD, QK, o);
        ok = rr.ok();
        for (uint32_t d = 0; d < HD && ok; d++) { ok &= close_float(s0[d], conv1[2 * QK * HD + d], 1e-6f, 1e-6f); }
        st.check(ok, "oracle gdn delta rule cancels retrieval error (beta=1)");
    }

    // -- gated norm: direct weight, norm BEFORE gate --------------------------
    {
        const uint32_t T = 2, H = 3, HD = 8;
        luisa::vector<float> o(T * H * HD), z(T * H * HD), w(HD), out;
        for (size_t i = 0; i < o.size(); i++) { o[i] = std::sin(i * 0.3f) * 2.0f; z[i] = std::cos(i * 0.2f) * 3.0f; }
        for (size_t i = 0; i < w.size(); i++) { w[i] = 0.5f + 0.1f * i; }
        auto r = ref_tile_gdn_gated_norm(o, z, w, T, H, HD, 1e-6f, out);
        bool ok = r.ok();
        // gate=0 (z very negative) must zero the output regardless of norm
        luisa::vector<float> zneg(z.size(), -50.0f), out2;
        auto r2 = ref_tile_gdn_gated_norm(o, zneg, w, T, H, HD, 1e-6f, out2);
        ok &= r2.ok();
        for (auto v : out2) { ok &= std::fabs(v) < 1e-6f; }
        st.check(ok, "oracle gdn gated norm (norm before gate)");
    }

    // -- capture validity of every factory (small static shapes) --------------
    {
        QwenTileConfig cfg;
        cfg.q_heads = 4; cfg.kv_heads = 2; cfg.attn_head_dim = 8; cfg.rope_dim = 4;
        cfg.gdn_qk_heads = 2; cfg.gdn_v_heads = 4; cfg.gdn_head_dim = 4;
        for (auto format : {QwenTileWeightFormat::bf16, QwenTileWeightFormat::fp8_e4m3}) {
            auto emb = capture_tile_embedding(format, 3, 128, 7);
            st.check_capture(emb, format == QwenTileWeightFormat::bf16 ? "embedding bf16" : "embedding fp8");
            TileGemmShape gs;
            // bf16 supports ragged even K (130); fp8 requires K % 4 == 0
            // (word-aligned rows), so use 132 for the fp8 ragged case.
            gs.m = 3; gs.n = 70; gs.k = format == QwenTileWeightFormat::bf16 ? 130 : 132;
            gs.tile_m = 16; gs.tile_n = 64;
            auto gemm = capture_tile_gemm(format, TileMatmulEpilogue::residual, gs);
            st.check_capture(gemm, format == QwenTileWeightFormat::bf16 ? "gemm bf16 ragged" : "gemm fp8 ragged");
            if (format == QwenTileWeightFormat::fp8_e4m3) {
                TileGemmShape bad = gs; bad.k = 130;// not divisible by 4
                auto rejected = capture_tile_gemm(format, TileMatmulEpilogue::none, bad);
                st.check(!rejected.ok(), "fp8 gemm K%4!=0 rejected");
            }
            TileGemmShape gv;
            gv.m = 1; gv.n = 65; gv.k = 64; gv.tile_m = 1; gv.tile_n = 64;
            auto gemv = capture_tile_gemm(format, TileMatmulEpilogue::none, gv);
            st.check_capture(gemv, format == QwenTileWeightFormat::bf16 ? "gemv bf16" : "gemv fp8");
        }
        auto norm = capture_tile_rmsnorm(TileNormEpilogue::plain, 5, 33, 1e-6f);
        st.check_capture(norm, "rmsnorm plain ragged");
        auto normr = capture_tile_rmsnorm(TileNormEpilogue::residual, 5, 33, 1e-6f);
        st.check_capture(normr, "rmsnorm residual");
        auto sw = capture_tile_swiglu_split(77);
        st.check_capture(sw, "swiglu split");
        auto swf = capture_tile_swiglu_fused(3, 19);
        st.check_capture(swf, "swiglu fused");
        auto add = capture_tile_residual_add(99);
        st.check_capture(add, "residual add");
        auto zero = capture_tile_zero_fill(99);
        st.check_capture(zero, "zero fill");
        auto prepq = capture_tile_attn_prepare_q(cfg, 3);
        st.check_capture(prepq, "attn prepare q");
        auto prepk = capture_tile_attn_prepare_k(cfg, 3);
        st.check_capture(prepk, "attn prepare k");
        auto prepv = capture_tile_attn_prepare_v(cfg, 3);
        st.check_capture(prepv, "attn prepare v");
        auto rope = capture_tile_rope_partial(3, 4, 8, 4);
        st.check_capture(rope, "rope partial (oracle)");
        auto ropet = capture_tile_rope_partial_table(3, 4, 8, 4, 9);
        st.check_capture(ropet, "rope partial table");
        auto app = capture_tile_kv_append(3, 16, 2, 9);
        st.check_capture(app, "kv append");
        auto pre = capture_tile_attn_prefill(cfg, 3, 2, 5, 9);
        st.check_capture(pre, "attn prefill");
        auto dec = capture_tile_attn_decode(cfg, 5, 9);
        st.check_capture(dec, "attn decode");
        for (uint32_t pm = 0; pm < 3; pm++) {
            auto conv = capture_tile_gdn_conv_silu_step(12, 4, pm);
            st.check_capture(conv, "gdn conv step");
        }
        auto convbad = capture_tile_gdn_conv_silu_step(12, 4, 3);
        st.check(!convbad.ok(), "gdn conv step pos_mod out of range rejected");
        auto gates = capture_tile_gdn_gates(3, 4, 20.0f);
        st.check_capture(gates, "gdn gates");
        auto qn = capture_tile_gdn_q_l2norm(3, 2, 4, 16, 1e-6f, 0.5f);
        st.check_capture(qn, "gdn q l2norm");
        auto kn = capture_tile_gdn_k_l2norm(3, 2, 4, 16, 1e-6f);
        st.check_capture(kn, "gdn k l2norm");
        auto rec = capture_tile_gdn_recurrent(3, 4, 2, 4);
        st.check_capture(rec, "gdn recurrent");
        auto gn = capture_tile_gdn_gated_norm(3, 4, 4, 1e-6f);
        st.check_capture(gn, "gdn gated norm");
        // model-scale captures (verifier only; no device compile here)
        QwenTileConfig full;
        auto prep_full_q = capture_tile_attn_prepare_q(full, 4);
        st.check_capture(prep_full_q, "attn prepare q model-scale");
        auto prep_full_k = capture_tile_attn_prepare_k(full, 4);
        st.check_capture(prep_full_k, "attn prepare k model-scale");
        auto prep_full_v = capture_tile_attn_prepare_v(full, 4);
        st.check_capture(prep_full_v, "attn prepare v model-scale");
        auto rope_full = capture_tile_rope_partial_table(4, full.q_heads, full.attn_head_dim, full.rope_dim, 256);
        st.check_capture(rope_full, "rope table model-scale");
        auto conv_full = capture_tile_gdn_conv_silu_step(full.gdn_qkv_n(), full.conv_kernel, 1);
        st.check_capture(conv_full, "gdn conv step model-scale");
        auto rec_full = capture_tile_gdn_recurrent(1, full.gdn_v_heads, full.gdn_qk_heads, full.gdn_head_dim);
        st.check_capture(rec_full, "gdn recurrent model-scale decode");
        TileGemmShape ff;
        ff.m = 1; ff.n = full.ffn; ff.k = full.hidden;
        auto gemm_full = capture_tile_gemm(QwenTileWeightFormat::fp8_e4m3, TileMatmulEpilogue::none, ff);
        st.check_capture(gemm_full, "gemm fp8 model-scale ffn");
    }

    // ------------------------------------------------------------------
    // EXTERNAL-CONTRACT FIXTURES (GGUF/HF semantics; independent hand-computed
    // expectations so a shared oracle/kernel bug cannot pass both sides).
    // Each fixture pins one converter/HF convention the kernels must honor.
    // ------------------------------------------------------------------
    // (a) RMS norm with KNOWN pre-added weights: w = [2, 0.5, 1, 1.5],
    // x = [3, 4, 0, -1] -> y = x * rsqrt(26/4 + eps) * w (DIRECT, no 1+w).
    {
        const luisa::vector<float> x{3.0f, 4.0f, 0.0f, -1.0f};
        const luisa::vector<float> w{2.0f, 0.5f, 1.0f, 1.5f};
        luisa::vector<float> y, sum;
        const auto r = ref_tile_rmsnorm(x, w, {}, false, 1u, 4u, 1e-6f, y, sum);
        const double inv = 1.0 / std::sqrt(26.0 / 4.0 + 1e-6);
        bool ok = r.ok();
        const float expect[4] = {float(3.0 * inv * 2.0), float(4.0 * inv * 0.5),
                                 0.0f, float(-1.0 * inv * 1.5)};
        for (uint32_t i = 0; ok && i < 4u; i++) { ok = close_float(y[i], expect[i], 1e-6f, 1e-7f); }
        st.check(ok, "fixture rmsnorm pre-added direct weight");
    }
    // (b) Gates with KNOWN negative ssm_a (GGUF -exp(A_log), no device exp):
    // ssm_a = [-2, -0.5], dt_bias = [0.25, -1], a = [1, 30], b = [0, 2].
    // g0 = -2 * log1p(exp(1.25)); g1 = -0.5 * 29 (softplus threshold 20).
    {
        const luisa::vector<float> a{1.0f, 30.0f}, b{0.0f, 2.0f};
        const luisa::vector<float> ssm_a{-2.0f, -0.5f}, dt{0.25f, -1.0f};
        luisa::vector<float> beta, g;
        const auto r = ref_tile_gdn_gates(a, b, ssm_a, dt, 1u, 2u, 20.0f, beta, g);
        const double g0 = -2.0 * std::log1p(std::exp(1.25));
        const double g1 = -0.5 * 29.0;
        const double b0 = 0.5, b1 = 1.0 / (1.0 + std::exp(-2.0));
        const bool ok = r.ok() &&
                        close_float(g[0], float(g0), 1e-6f, 1e-7f) &&
                        close_float(g[1], float(g1), 1e-6f, 1e-7f) &&
                        close_float(beta[0], float(b0), 1e-6f, 1e-7f) &&
                        close_float(beta[1], float(b1), 1e-6f, 1e-7f);
        st.check(ok, "fixture gdn gates negative ssm_a direct");
    }
    // (c) Channel-major impulse conv: C=2, K=4, weight[c*K+j] = c*10+j+1.
    // src = impulse at t=0 on channel 0 only. Causal depthwise conv (tap j
    // reads position p-(K-1)+j): out[t,0] = silu(w[0*K + (K-1-t)]) for
    // t in [0,K), zero afterwards; channel 1 stays zero. A kernel-major
    // misread would use w[(K-1-t)*C + 0] and fail. Ring history zero.
    {
        const uint32_t C = 2u, K = 4u, T = 4u;
        luisa::vector<float> w(C * K), src(T * C, 0.0f), ring(C * (K - 1u), 0.0f);
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t j = 0; j < K; j++) { w[c * K + j] = float(c * 10 + j + 1); }
        }
        src[0u * C + 0u] = 1.0f;// impulse: token 0, channel 0
        luisa::vector<float> out;
        const auto r = ref_tile_gdn_conv(src, w, ring, T, C, K, 0u, out);
        bool ok = r.ok();
        for (uint32_t t = 0; t < T && ok; t++) {
            for (uint32_t c = 0; c < C && ok; c++) {
                const double acc = (c == 0u && t < K) ? static_cast<double>(w[0u * K + (K - 1u - t)]) : 0.0;
                ok = close_float(out[t * C + c], float(host_silu(acc)), 1e-6f, 1e-7f);
            }
        }
        st.check(ok, "fixture conv channel-major impulse");
    }
    // (d) Tiled V-head pairing: qk_heads=2, v_heads=4, hd=32. Distinct unit
    // q/k per QK head, distinct v per V head, beta=1, g=0, zero state.
    // After one token S = k v^T so o[vh] = (q[qk] . k[qk]) * v[vh];
    // tiled mapping requires qk = vh % 2 (grouped vh/2 would differ).
    {
        const uint32_t QK = 2u, VH = 4u, HD = 32u;
        luisa::vector<float> q(QK * HD, 0.0f), k(QK * HD, 0.0f);
        q[0] = 1.0f; q[HD + 1] = 1.0f;
        k[0] = 1.0f; k[HD + 1] = 1.0f;
        const uint32_t C = 2u * QK * HD + VH * HD;
        luisa::vector<float> conv(C, 0.0f);
        for (uint32_t vh = 0; vh < VH; vh++) { conv[2u * QK * HD + vh * HD + 0u] = float(vh + 1); }
        luisa::vector<float> beta(VH, 1.0f), g(VH, 0.0f), state(VH * HD * HD, 0.0f), out;
        const auto r = ref_tile_gdn_recurrent(q, k, conv, beta, g, state, 1u, VH, HD, QK, out);
        bool ok = r.ok();
        for (uint32_t vh = 0; vh < VH && ok; vh++) {
            ok = close_float(out[vh * HD + 0u], float(vh + 1), 1e-6f, 1e-7f);
            for (uint32_t i = 1; i < HD && ok; i++) { ok = out[vh * HD + i] == 0.0f; }
        }
        st.check(ok, "fixture gdn tiled v-head pairing (h % qk_heads)");
    }
    // (e) NeoX rotate-half rope: hd=8, rope_dim=4 (half=2), theta=10000,
    // pos=1, x=[1,2,3,4,5,6,7,8]. freq[p]=10000^(-2p/4).
    // out[0]=c0-3s0; out[2]=s0+3c0; out[1]=2c1-4s1; out[3]=2s1+4c1;
    // tail [4..8) passthrough.
    {
        const luisa::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
        const luisa::vector<int32_t> pos{1};
        luisa::vector<float> out;
        const auto r = ref_tile_rope_partial(x, pos, 1u, 1u, 8u, 4u, 10000.0f, out);
        const double f0 = 1.0, f1 = std::pow(10000.0, -0.5);
        const double c0 = std::cos(f0), s0 = std::sin(f0);
        const double c1 = std::cos(f1), s1 = std::sin(f1);
        const float expect[8] = {float(1 * c0 - 3 * s0), float(2 * c1 - 4 * s1),
                                 float(1 * s0 + 3 * c0), float(2 * s1 + 4 * c1),
                                 5, 6, 7, 8};
        bool ok = r.ok();
        for (uint32_t i = 0; ok && i < 8u; i++) { ok = close_float(out[i], expect[i], 1e-5f, 1e-6f); }
        st.check(ok, "fixture rope NeoX rotate-half + tail");
    }
    // (f) Near-zero L2 rows: conv row q = [1e-4, 0...], k = [0...].
    // FLA l2norm: q_out = x * rsqrt(sum+eps) * q_scale (tiny, NOT ~1/eps
    // blown up); k_out exactly 0.
    {
        const uint32_t QK = 1u, HD = 32u, VN = 32u;
        const uint32_t C = 2u * QK * HD + VN;
        luisa::vector<float> conv(C, 0.0f);
        conv[0] = 1e-4f;
        luisa::vector<float> q, k;
        const float qs = 1.0f / std::sqrt(32.0f);
        const auto r = ref_tile_gdn_qk_norm(conv, 1u, QK, HD, VN, 1e-6f, qs, q, k);
        const double inv = 1.0 / std::sqrt(1e-8 + 1e-6);
        bool ok = r.ok() && close_float(q[0], float(1e-4 * inv * qs), 1e-5f, 1e-9f);
        for (uint32_t i = 1; i < HD && ok; i++) { ok = q[i] == 0.0f; }
        for (uint32_t i = 0; i < HD && ok; i++) { ok = k[i] == 0.0f; }
        st.check(ok, "fixture gdn l2norm near-zero rows (eps inside rsqrt)");
    }

    // -- fail-closed validation ------------------------------------------------
    {
        auto bad = capture_tile_gemm(QwenTileWeightFormat::bf16, TileMatmulEpilogue::none, {1, 8, 3, 1, 8});
        st.check(!bad.ok() && bad.status.code == TileOpStatus::invalid_argument, "bf16 odd-K rejected");
        auto bad2 = capture_tile_kv_append(4, 8, 6, 9);
        st.check(!bad2.ok(), "kv append capacity overflow rejected");
        auto bad3 = capture_tile_attn_decode(QwenTileConfig{}, 0, 9);
        st.check(!bad3.ok(), "decode kv_len=0 rejected");
        auto chunk = tile_ops_chunk_prefill_status();
        st.check(!chunk.ok() && chunk.code == TileOpStatus::unsupported, "chunk prefill gap reported");
    }

    if (!st.all_ok) {
        return TileOpResult::error(TileOpStatus::capture_failed, "tile_ops self-test failed (see report)");
    }
    return TileOpResult::success();
}

}// namespace luisa::qwen::tile_ops
