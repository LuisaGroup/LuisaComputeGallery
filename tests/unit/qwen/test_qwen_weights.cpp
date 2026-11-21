// Test for src/qwen/qwen/common.h/.cpp + weights.h/.cpp -- host codecs,
// storage classification, host packers, and (when a backend is given as
// argv[1]) device upload round-trips.
// Host-only by default: no device, no backend, no exceptions.
//
// Covers:
// - BF16 RNE codec: exact values, ties-to-even, overflow to inf, NaN,
//   subnormals, round-trip over a dense sample grid;
// - FP8 E4M3FN codec: all 256 codes decode finitely (except the two NaN
//   codes), max 448, subnormals, RNE ties, saturation, NaN handling,
//   encode(decode(c)) == c for every finite code;
// - packed pair/quad ABI: low element first, zero padding, size validation;
// - per-row group-of-64 FP8 quantization: scale = max_abs/448, zero groups,
//   ragged K (not a multiple of 64), saturation, dequantize reference;
// - storage classification: norms/conv/a/dt stay F32, BF16 matrices pack,
//   Q8_0 explicitly rejected (never silently FP8);
// - device upload (optional, argv[1] = backend): the REAL
//   QwenDeviceWeights::upload over a synthetic GGUF that satisfies the full
//   frozen Qwen3.5-4B inventory (426 tensors, ~3.9M elements; no real
//   checkpoint needed). Tiny chunk_bytes exercise the streaming paths.
//   Both variants (BF16 packed and FP8 packed) round-trip every tensor
//   against the host reference; report counts and fp8_quantized diagnostics
//   are asserted.
#include "ut/ut.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

#include <luisa/core/stl/filesystem.h>

#include "qwen/common.h"
#include "qwen/dsl_ops.h"
#include "qwen/gguf.h"
#include "qwen/model.h"
#include "qwen/w4_pack.h"
#include "qwen/projection_pack.h"
#include "qwen/weights.h"

#include <luisa/runtime/context.h>

using namespace luisa;
using namespace luisa::qwen;
using namespace boost::ut;
using namespace boost::ut::literals;

int g_argc = 0;
const char **g_argv = nullptr;

namespace {

struct TempDir {
    luisa::filesystem::path path;
    TempDir() {
        path = luisa::filesystem::temp_directory_path() /
               ("lcqwen_w_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
        luisa::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        luisa::filesystem::remove_all(path, ec);
    }
};

void put_u32(luisa::vector<std::byte> &v, uint32_t x) {
    for (uint32_t i = 0; i < 4u; ++i) { v.emplace_back(static_cast<std::byte>((x >> (8u * i)) & 0xffu)); }
}
void put_u64(luisa::vector<std::byte> &v, uint64_t x) {
    for (uint32_t i = 0; i < 8u; ++i) { v.emplace_back(static_cast<std::byte>((x >> (8u * i)) & 0xffu)); }
}
void put_str(luisa::vector<std::byte> &v, luisa::string_view s) {
    put_u64(v, s.size());
    for (auto c : s) { v.emplace_back(static_cast<std::byte>(c)); }
}
void put_f32(luisa::vector<std::byte> &v, float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4u);
    put_u32(v, bits);
}
void put_f64(luisa::vector<std::byte> &v, double x) {
    uint64_t bits;
    std::memcpy(&bits, &x, 8u);
    put_u64(v, bits);
}

// One metadata key/value (GGUF v3 value types: 4=u32, 6=f32, 8=string, 9=array).
struct MetaKV {
    luisa::string key;
    uint32_t type;
    uint64_t u = 0;
    double f = 0;
    luisa::string s;
    luisa::vector<uint64_t> arr;// u32 array elements
};
void put_meta(luisa::vector<std::byte> &v, const MetaKV &kv) {
    put_str(v, kv.key);
    put_u32(v, kv.type);
    switch (kv.type) {
        case 4u: put_u32(v, static_cast<uint32_t>(kv.u)); break;
        case 6u: put_f32(v, static_cast<float>(kv.f)); break;
        case 8u: put_str(v, kv.s); break;
        case 9u:
            put_u32(v, 4u);// element type u32
            put_u64(v, kv.arr.size());
            for (auto e : kv.arr) { put_u32(v, static_cast<uint32_t>(e)); }
            break;
        default: break;
    }
}

// Build a synthetic GGUF with metadata and the given tensors (F32=0 / BF16=30).
luisa::vector<std::byte> make_gguf(
    const luisa::vector<std::tuple<luisa::string, uint32_t, luisa::vector<uint64_t>, luisa::vector<std::byte>>> &tensors,
    const luisa::vector<MetaKV> &meta = {}) {
    luisa::vector<std::byte> out;
    put_u32(out, 0x46554747u);
    put_u32(out, 3u);
    put_u64(out, tensors.size());
    put_u64(out, meta.size());
    for (const auto &kv : meta) { put_meta(out, kv); }
    // compute relative offsets (32-aligned per tensor)
    luisa::vector<uint64_t> offsets;
    uint64_t rel = 0;
    for (const auto &[name, type, dims, payload] : tensors) {
        offsets.emplace_back(rel);
        rel += payload.size();
        while (rel % 32u != 0u) { ++rel; }
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &[name, type, dims, payload] = tensors[i];
        put_str(out, name);
        put_u32(out, static_cast<uint32_t>(dims.size()));
        for (auto d : dims) { put_u64(out, d); }
        put_u32(out, type);
        put_u64(out, offsets[i]);
    }
    while (out.size() % 32u != 0u) { out.emplace_back(std::byte{0}); }
    const uint64_t data_start = out.size();
    for (size_t i = 0; i < tensors.size(); ++i) {
        const uint64_t abs = data_start + offsets[i];
        while (out.size() < abs) { out.emplace_back(std::byte{0}); }
        const auto &payload = std::get<3>(tensors[i]);
        out.insert(out.end(), payload.begin(), payload.end());
    }
    return out;
}

void write_file(const luisa::filesystem::path &p, luisa::span<const std::byte> data) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
}

luisa::vector<std::byte> f32_bytes(luisa::span<const float> v) {
    luisa::vector<std::byte> b(v.size_bytes());
    std::memcpy(b.data(), v.data(), v.size_bytes());
    return b;
}
luisa::vector<std::byte> bf16_bytes(luisa::span<const float> v) {
    luisa::vector<std::byte> b(v.size() * 2u);
    for (size_t i = 0; i < v.size(); ++i) {
        const uint16_t c = f32_to_bf16(v[i]);
        b[2u * i] = static_cast<std::byte>(c & 0xffu);
        b[2u * i + 1u] = static_cast<std::byte>(c >> 8u);
    }
    return b;
}

// ---------------------------------------------------------------------------
// Codec tests
// ---------------------------------------------------------------------------
void test_bf16_codec() {
    // exact representable values
    expect(bf16_to_f32(0x3f80u) == 1.0f);
    expect(bf16_to_f32(0xbf80u) == -1.0f);
    expect(bf16_to_f32(0x4000u) == 2.0f);
    expect(bf16_to_f32(0x0000u) == 0.0f);
    expect(f32_to_bf16(1.0f) == 0x3f80u);
    expect(f32_to_bf16(-2.0f) == 0xc000u);
    expect(f32_to_bf16(0.0f) == 0x0000u);
    // sign of zero preserved
    expect(f32_to_bf16(-0.0f) == 0x8000u);
    // ties-to-even: 1.0 + 2^-8 is exactly halfway between 1.0 and 1.0078125
    const float tie = 1.0f + 0x1p-8f;
    expect(f32_to_bf16(tie) == 0x3f80u);// rounds down to even
    const float tie2 = 1.0078125f + 0x1p-8f;
    expect(f32_to_bf16(tie2) == 0x3f82u);// rounds up to even
    // overflow to inf
    expect(f32_to_bf16(3.4e38f) == 0x7f80u);
    expect(std::isinf(bf16_to_f32(0x7f80u)));
    // NaN stays NaN
    expect(std::isnan(bf16_to_f32(f32_to_bf16(std::nanf("")))));
    // round-trip over a grid of finite values
    for (int i = -1000; i <= 1000; ++i) {
        const float v = static_cast<float>(i) * 0.0137f;
        const float rt = bf16_to_f32(f32_to_bf16(v));
        const float err = std::abs(rt - v);
        expect(err <= std::abs(v) * 0x1p-7f + 1e-30f);
    }
}

void test_fp8_codec() {
    // decode table spot checks
    expect(fp8_e4m3_to_f32(0x00u) == 0.0f);
    expect(fp8_e4m3_to_f32(0x80u) == -0.0f);
    expect(fp8_e4m3_to_f32(0x38u) == 1.0f);   // exp 7, mant 0
    expect(fp8_e4m3_to_f32(0x7eu) == 448.0f); // max finite
    expect(fp8_e4m3_to_f32(0xfeu) == -448.0f);
    expect(std::isnan(fp8_e4m3_to_f32(0x7fu)));
    expect(std::isnan(fp8_e4m3_to_f32(0xffu)));
    expect(fp8_e4m3_to_f32(0x01u) == 0x1p-9f); // min subnormal
    expect(fp8_e4m3_to_f32(0x08u) == 0x1p-6f); // min normal
    // every finite code round-trips through the encoder
    for (uint32_t c = 0; c < 256u; ++c) {
        if (c == 0x7fu || c == 0xffu) { continue; }
        const float v = fp8_e4m3_to_f32(static_cast<uint8_t>(c));
        expect(std::isfinite(v));
        const uint8_t back = f32_to_fp8_e4m3(v);
        expect(back == static_cast<uint8_t>(c));
    }
    // saturation
    expect(f32_to_fp8_e4m3(1000.0f) == 0x7eu);
    expect(f32_to_fp8_e4m3(-1000.0f) == 0xfeu);
    expect(f32_to_fp8_e4m3(INFINITY) == 0x7eu);
    expect(f32_to_fp8_e4m3(-INFINITY) == 0xfeu);
    // NaN -> quiet NaN code
    expect(f32_to_fp8_e4m3(std::nanf("")) == kFp8E4M3Nan);
    // RNE tie: halfway between 1.0 (0x38) and 1.125 (0x39) is 1.0625 -> even (0x38)
    expect(f32_to_fp8_e4m3(1.0625f) == 0x38u);
    // halfway between 1.125 (0x39) and 1.25 (0x3a) is 1.1875 -> even (0x3a)
    expect(f32_to_fp8_e4m3(1.1875f) == 0x3au);
    // tiny values below half the min subnormal round to zero
    expect(f32_to_fp8_e4m3(0x1p-11f) == 0x00u);
    // exact half of min subnormal ties to even (zero)
    expect(f32_to_fp8_e4m3(0x1p-10f) == 0x00u);
    // just above half rounds up to min subnormal
    expect(f32_to_fp8_e4m3(0x1p-10f + 0x1p-30f) == 0x01u);
}

void test_packing() {
    // BF16 pairs: low element first
    const uint16_t codes[5] = {0x1111u, 0x2222u, 0x3333u, 0x4444u, 0x5555u};
    uint32_t packed[3];
    expect(pack_bf16_pairs(codes, packed).ok());
    expect(packed[0] == 0x22221111u);
    expect(packed[1] == 0x44443333u);
    expect(packed[2] == 0x00005555u);// zero-padded high half
    uint16_t back[5];
    expect(unpack_bf16_pairs(packed, 5u, back).ok());
    for (int i = 0; i < 5; ++i) { expect(back[i] == codes[i]); }
    // nonzero padding rejected
    packed[2] = 0xdead5555u;
    expect(!unpack_bf16_pairs(packed, 5u, back).ok());
    // size mismatch rejected
    expect(!pack_bf16_pairs(codes, luisa::span<uint32_t>{packed, 2}).ok());
    // FP8 quads: low element first
    const uint8_t fcodes[6] = {0x01u, 0x02u, 0x03u, 0x04u, 0x05u, 0x06u};
    uint32_t fpacked[2];
    expect(pack_fp8_quads(fcodes, fpacked).ok());
    expect(fpacked[0] == 0x04030201u);
    expect(fpacked[1] == 0x00000605u);
    uint8_t fback[6];
    expect(unpack_fp8_quads(fpacked, 6u, fback).ok());
    for (int i = 0; i < 6; ++i) { expect(fback[i] == fcodes[i]); }
    fpacked[1] = 0x00ff0605u;
    expect(!unpack_fp8_quads(fpacked, 6u, fback).ok());
}

void test_fp8_quantize() {
    // Ragged row: K = 130 -> 3 groups (64, 64, 2).
    constexpr uint64_t K = 130u;
    float values[K];
    for (uint64_t i = 0; i < K; ++i) {
        values[i] = (i % 7u == 0u) ? 0.0f : static_cast<float>(i) * 0.01f - 0.5f;
    }
    uint8_t codes[K];
    float scales[3];
    expect(quantize_fp8_row(values, codes, scales).ok());
    // group 0 max_abs
    float m0 = 0.f;
    for (uint64_t i = 0; i < 64u; ++i) { m0 = std::max(m0, std::abs(values[i])); }
    expect(std::abs(scales[0] - m0 / 448.0f) <= 1e-9f);
    // dequantize reference error bound: |dq - v| <= scale * 0.5 * step
    for (uint64_t g = 0; g < 3u; ++g) {
        const uint64_t begin = g * 64u;
        const uint64_t end = std::min<uint64_t>(begin + 64u, K);
        for (uint64_t i = begin; i < end; ++i) {
            const float dq = fp8_e4m3_to_f32(codes[i]) * scales[g];
            // E4M3 relative precision is 2^-3 (3 mantissa bits) -> half-ulp ~ 1/16
            expect(std::abs(dq - values[i]) <= std::abs(values[i]) * 0.0625f + scales[g] * 0.5f);
        }
    }
    // zero group
    float z[64] = {};
    uint8_t zc[64];
    float zs[1];
    expect(quantize_fp8_row(z, zc, zs).ok());
    expect(zs[0] == 0.0f);
    for (int i = 0; i < 64; ++i) { expect(zc[i] == 0u); }
    // non-finite input rejected
    float bad[64] = {};
    bad[3] = INFINITY;
    expect(!quantize_fp8_row(bad, zc, zs).ok());
    // matrix round-trip with ragged K and multiple rows
    constexpr uint64_t N = 3u;
    float mat[K * N];
    for (uint64_t i = 0; i < K * N; ++i) { mat[i] = std::sin(static_cast<float>(i) * 0.1f); }
    luisa::vector<uint32_t> packed((K * N + 3u) / 4u);
    luisa::vector<float> mscales(N * 3u);
    expect(quantize_fp8_matrix(mat, K, N, packed, mscales).ok());
    luisa::vector<float> dq(K * N);
    expect(dequantize_fp8_matrix(packed, mscales, K, N, dq).ok());
    for (uint64_t row = 0; row < N; ++row) {
        for (uint64_t i = 0; i < K; ++i) {
            const uint64_t g = i / 64u;
            const float sc = mscales[row * 3u + g];
            const uint64_t gi = row * K + i;
            const uint8_t code = static_cast<uint8_t>((packed[gi / 4u] >> (8u * (gi % 4u))) & 0xffu);
            // ulp at the code's own binade (next code up crosses binades at
            // mantissa 7, so use the spacing below the code, doubled there).
            const float below = fp8_e4m3_to_f32(code == 0u ? 0u : static_cast<uint8_t>(code - 1u));
            const float here = fp8_e4m3_to_f32(code);
            float step = here - below;
            if ((code & 0x7u) == 0u && code != 0u) { step *= 2.0f; }// binade start
            // error <= half an E4M3 ulp at that code, times the group scale
            const float bound = sc * (std::abs(step) * 0.5f + 1e-6f);
            expect(std::abs(dq[gi] - mat[gi]) <= bound + 1e-6f);
        }
    }
    // wrong sizes rejected
    expect(!quantize_fp8_matrix(mat, K, N, luisa::span<uint32_t>{packed.data(), 1}, mscales).ok());
}

void test_classification() {
    const uint64_t d2[] = {2560u, 9216u};
    const uint64_t d3[] = {4u, 8192u};
    const uint64_t d4[] = {32u};
    // F32 always dense
    const uint64_t d1[] = {2560u};
    auto a = classify_weight_storage("blk.0.attn_norm.weight", 0u, d1, false);
    expect(a.ok() && *a.value == WeightStorage::f32_dense);
    // BF16 matrix -> packed (or fp8)
    auto b = classify_weight_storage("blk.0.ffn_gate.weight", 30u, d2, false);
    expect(b.ok() && *b.value == WeightStorage::bf16_packed);
    auto c = classify_weight_storage("blk.0.ffn_gate.weight", 30u, d2, true);
    expect(c.ok() && *c.value == WeightStorage::fp8_packed);
    // conv stays dense even if BF16
    auto d = classify_weight_storage("blk.0.ssm_conv1d.weight", 30u, d3, true);
    expect(d.ok() && *d.value == WeightStorage::f32_dense);
    // 1-D stays dense
    auto e = classify_weight_storage("blk.0.ssm_a", 30u, d4, true);
    expect(e.ok() && *e.value == WeightStorage::f32_dense);
    // Q8_0 explicitly rejected without the native opt-in ...
    auto q = classify_weight_storage("blk.0.ffn_gate.weight", 8u, d2, true);
    expect(!q.ok());
    expect(q.status.code == StatusCode::unsupported);
    expect(q.status.message.find("Q8_0") != luisa::string::npos);
    auto qbf = classify_weight_storage("blk.0.ffn_gate.weight", 8u, d2, false);
    expect(!qbf.ok());// the default (bf16) variant refuses Q8_0 too
    // ... and ACCEPTED as the native block format when the caller opts in.
    auto qok = classify_weight_storage("blk.0.ffn_gate.weight", 8u, d2, false, true);
    expect(qok.ok() && *qok.value == WeightStorage::q8_0_packed);
    expect(weight_storage_name(WeightStorage::q8_0_packed) == "q8_0_packed");
    // Q8_0 geometry: K % 32 != 0 is refused (whole blocks only).
    const uint64_t dq_bad[] = {2540u, 64u};
    auto qbad = classify_weight_storage("blk.0.ffn_gate.weight", 8u, dq_bad, false, true);
    expect(!qbad.ok());
    expect(qbad.status.code == StatusCode::unsupported);
    const uint64_t dq_zero[] = {0u, 64u};
    expect(!classify_weight_storage("blk.0.ffn_gate.weight", 8u, dq_zero, false, true).ok());
    // A Q8_0 sensitive tensor (1-D norm / conv) is refused, never dequantized.
    const uint64_t d1q[] = {2560u};
    auto qnorm = classify_weight_storage("blk.0.attn_norm.weight", 8u, d1q, false, true);
    expect(!qnorm.ok());
    const uint64_t convq[] = {4u, 8192u};
    expect(!classify_weight_storage("blk.0.ssm_conv1d.weight", 8u, convq, false, true).ok());
    // FP8 and native Q8_0 are mutually exclusive (never routed through each other).
    auto both = classify_weight_storage("blk.0.ffn_gate.weight", 8u, d2, true, true);
    expect(!both.ok());
    expect(both.status.code == StatusCode::invalid_argument);
    // Scale/group geometry helpers.
    expect(q8_0_group_count(2560u) == 80u);
    expect(q8_0_group_count(9216u) == 288u);
    // other quant types rejected
    auto u = classify_weight_storage("blk.0.ffn_gate.weight", 12u, d2, false);
    expect(!u.ok());
}

// ---------------------------------------------------------------------------
// Device upload round-trip (only when a backend is passed as argv[1]).
// Exercises the REAL QwenDeviceWeights::upload: the synthetic GGUF below
// satisfies the full frozen Qwen3.5-4B contract (metadata + the exact
// 426-tensor inventory), so no real checkpoint is required. Values are small
// deterministic patterns; chunk_bytes is tiny (clamped to 64 KiB) so the
// streaming paths run many chunks per tensor.
// ---------------------------------------------------------------------------
luisa::vector<MetaKV> qwen35_meta() {
    // Frozen metadata values from src/qwen/qwen/model.h (Qwen35Geometry).
    return {
        {"general.architecture", 8u, 0, 0, "qwen35", {}},
        {"qwen35.block_count", 4u, 32u, 0, "", {}},
        {"qwen35.context_length", 4u, 262144u, 0, "", {}},
        {"qwen35.embedding_length", 4u, 2560u, 0, "", {}},
        {"qwen35.feed_forward_length", 4u, 9216u, 0, "", {}},
        {"qwen35.attention.head_count", 4u, 16u, 0, "", {}},
        {"qwen35.attention.head_count_kv", 4u, 4u, 0, "", {}},
        {"qwen35.attention.key_length", 4u, 256u, 0, "", {}},
        {"qwen35.attention.value_length", 4u, 256u, 0, "", {}},
        {"qwen35.rope.dimension_count", 4u, 64u, 0, "", {}},
        {"qwen35.ssm.conv_kernel", 4u, 4u, 0, "", {}},
        {"qwen35.ssm.state_size", 4u, 128u, 0, "", {}},
        {"qwen35.ssm.group_count", 4u, 16u, 0, "", {}},
        {"qwen35.ssm.time_step_rank", 4u, 32u, 0, "", {}},
        {"qwen35.ssm.inner_size", 4u, 4096u, 0, "", {}},
        {"qwen35.full_attention_interval", 4u, 4u, 0, "", {}},
        {"qwen35.rope.freq_base", 6u, 0, 1e7, "", {}},
        {"qwen35.attention.layer_norm_rms_epsilon", 6u, 0, 1e-6, "", {}},
        {"qwen35.rope.dimension_sections", 9u, 0, 0, "", {11u, 11u, 10u, 0u}},
        {"tokenizer.ggml.model", 8u, 0, 0, "gpt2", {}},
        {"tokenizer.ggml.pre", 8u, 0, 0, "qwen35", {}},
        {"tokenizer.ggml.eos_token_id", 4u, 248044u, 0, "", {}},
        {"tokenizer.ggml.padding_token_id", 4u, 248044u, 0, "", {}},
    };
}

// ---------------------------------------------------------------------------
// Synthetic Q8_0 block payloads (phase 8a). These build a Q8_0 GGUF tensor the
// way a real quantizer would: signed int8 codes + one IEEE FP16 scale per
// 32-element block. The uploader must copy the codes VERBATIM and widen the
// scale exactly, so the round-trip below is exact equality with int8*scale --
// not a tolerance, and not a re-quantization of the BF16 source.
// ---------------------------------------------------------------------------
uint16_t q8_0_scale_bits(uint64_t block) {
    // Positive, exactly-representable FP16 scales in [2^-12, 2^-9]: the exact
    // value the loader must widen is fp16_to_f32() of these bits.
    return f32_to_fp16(static_cast<float>(1u + (block % 7u)) / 8192.0f);
}

int8_t q8_0_code(uint32_t salt, uint64_t i) {
    // Full signed range, deterministic, plus the -128/127 extremes on purpose.
    const auto pattern = static_cast<uint32_t>(i * 37u + salt * 11u);
    if ((pattern & 0x3FFu) == 0u) { return -128; }
    if ((pattern & 0x3FFu) == 1u) { return 127; }
    if ((pattern & 0x3FFu) == 2u) { return 0; }
    return static_cast<int8_t>(static_cast<int>(pattern % 256u) - 128);
}

// 34 bytes per 32 codes: little-endian FP16 scale, then 32 signed int8 codes.
luisa::vector<std::byte> q8_0_bytes(luisa::span<const int8_t> codes) {
    luisa::vector<std::byte> out;
    const uint64_t blocks = static_cast<uint64_t>(codes.size()) / kQ8_0BlockSize;
    out.reserve(static_cast<size_t>(blocks * kQ8_0BlockBytes));
    for (uint64_t b = 0; b < blocks; ++b) {
        const uint16_t bits = q8_0_scale_bits(b);
        out.emplace_back(static_cast<std::byte>(bits & 0xffu));
        out.emplace_back(static_cast<std::byte>(bits >> 8u));
        for (uint64_t i = 0; i < kQ8_0BlockSize; ++i) {
            out.emplace_back(static_cast<std::byte>(
                static_cast<uint8_t>(codes[static_cast<size_t>(b * kQ8_0BlockSize + i)])));
        }
    }
    return out;
}

// Deterministic synthetic value for tensor `salt`, element i. Small magnitudes
// keep BF16 rounding exact-ish and FP8 quantization well-conditioned; the
// per-tensor salt makes cross-tensor mix-ups detectable.
float synth_value(uint32_t salt, uint64_t i) {
    return std::sin(static_cast<float>(i % 4096u) * 0.017f + static_cast<float>(salt) * 0.31f) * 0.25f;
}


// ---------------------------------------------------------------------------
// W4A16 host contract (opt_plan/05 Q2): geometry, deterministic clipping search,
// exact decode, reproducibility, NaN/overflow handling, and the dsl-layer mirror.
// ---------------------------------------------------------------------------
void test_w4_geometry() {
    // Accepted shapes: the plan's default group 128 plus the 64/32 variants.
    expect(validate_w4_geometry(2560u, 128u).ok());
    expect(validate_w4_geometry(4096u, 128u).ok());
    expect(validate_w4_geometry(9216u, 128u).ok());
    expect(validate_w4_geometry(1024u, 64u).ok());
    expect(validate_w4_geometry(2560u, 32u).ok());
    // K must be a whole number of 8-code words.
    expect(!validate_w4_geometry(2562u, 128u).ok());
    expect(!validate_w4_geometry(2540u, 128u).ok());
    // Ragged groups are refused, never padded.
    expect(!validate_w4_geometry(2560u, 96u).ok());
    expect(!validate_w4_geometry(2600u, 128u).ok());
    // Zero K and an unknown group are refused.
    expect(!validate_w4_geometry(0u, 128u).ok());
    expect(!validate_w4_geometry(2560u, 0u).ok());
    // The clip-search config itself is validated.
    W4ClipSearchConfig cfg{};
    expect(cfg.validate().ok());
    expect(!W4ClipSearchConfig{128u, 0.0f, 1.0f, 31u}.validate().ok());
    expect(!W4ClipSearchConfig{128u, 0.9f, 1.1f, 31u}.validate().ok());
    expect(!W4ClipSearchConfig{128u, 1.0f, 0.9f, 31u}.validate().ok());
    expect(!W4ClipSearchConfig{128u, 0.85f, 1.0f, 1u}.validate().ok());
    expect(!W4ClipSearchConfig{48u, 0.85f, 1.0f, 31u}.validate().ok());
    // Ratio grid is exact at both ends (deterministic, no RNG).
    expect(cfg.ratio_at(0u) == static_cast<double>(cfg.min_ratio));
    expect(cfg.ratio_at(30u) == 1.0);
    // int4 decode is the exact two's-complement sign extension.
    expect(w4_code_to_f32(0x0u) == 0.0f);
    expect(w4_code_to_f32(0x7u) == 7.0f);
    expect(w4_code_to_f32(0x8u) == -8.0f);
    expect(w4_code_to_f32(0xFu) == -1.0f);
    // Pack storage classification: matrices pack, sensitive tensors stay dense.
    const uint64_t dmat[] = {2560u, 9216u};
    const uint64_t dnorm[] = {2560u};
    const uint64_t dconv[] = {4u, 8192u};
    auto m = classify_w4_pack_storage("blk.0.ffn_gate.weight", dmat, 128u);
    expect(m.ok() && *m.value == WeightStorage::w4_packed);
    auto n = classify_w4_pack_storage("blk.0.attn_norm.weight", dnorm, 128u);
    expect(n.ok() && *n.value == WeightStorage::f32_dense);
    auto cv = classify_w4_pack_storage("blk.0.ssm_conv1d.weight", dconv, 128u);
    expect(cv.ok() && *cv.value == WeightStorage::f32_dense);
    const uint64_t dbad[] = {2540u, 64u};
    expect(!classify_w4_pack_storage("blk.0.ffn_gate.weight", dbad, 128u).ok());
    expect(weight_storage_name(WeightStorage::w4_packed) == "w4_packed");
}

void test_w4_pack_roundtrip() {
    // A matrix whose per-group max-abs varies by an order of magnitude, so the
    // clipping search has something to find and the group scales differ.
    constexpr uint64_t K = 256u;// two 128-groups
    constexpr uint64_t N = 4u;
    luisa::vector<float> w(K * N);
    for (uint64_t n = 0; n < N; ++n) {
        for (uint64_t k = 0; k < K; ++k) {
            const float scale = (k < 128u) ? 0.031f : 0.0017f;
            // A deterministic, non-uniform pattern with outliers at the ends of a
            // group (exactly the shape clipping is meant to help with).
            const float v = std::sin(static_cast<float>(k) * 0.37f + static_cast<float>(n)) * scale;
            w[n * K + k] = (k == 3u || k == 131u) ? 3.0f * scale : v;
        }
    }
    W4ClipSearchConfig cfg{};
    cfg.group = 128u;
    luisa::vector<uint32_t> words((K * N) / 8u, 0u);
    luisa::vector<float> scales(N * (K / 128u), 0.0f);
    W4PackStats stats{};
    expect(pack_w4_matrix(w, K, N, cfg, words, scales, &stats).ok());
    expect(stats.groups == N * 2u);
    expect(stats.max_abs > 0.0);
    // The clipping search must beat plain RTN (the whole point of the item); the
    // measured gain on this synthetic pattern is large because of the outliers.
    expect(stats.mse <= stats.rtn_mse);
    expect(stats.mean_ratio <= 1.0f);
    expect(stats.clipped_groups > 0u);
    // Reproducibility: a second call produces byte-identical words and scales.
    luisa::vector<uint32_t> words2((K * N) / 8u, 0u);
    luisa::vector<float> scales2(N * (K / 128u), 0.0f);
    expect(pack_w4_matrix(w, K, N, cfg, words2, scales2, nullptr).ok());
    expect(std::memcmp(words.data(), words2.data(), words.size() * 4u) == 0);
    expect(std::memcmp(scales.data(), scales2.data(), scales.size() * 4u) == 0);
    // Exact decode: dequantize() == code*scale with only the representable codes.
    luisa::vector<float> dq(K * N, 0.0f);
    expect(dequantize_w4_matrix(words, scales, K, N, 128u, dq).ok());
    for (uint64_t n = 0; n < N; ++n) {
        for (uint64_t k = 0; k < K; ++k) {
            const uint64_t g = k / 128u;
            const float sc = scales[n * 2u + g];
            const uint32_t nibble = (words[(n * K + k) / 8u] >> (4u * ((n * K + k) % 8u))) & 0xFu;
            expect(dq[n * K + k] == w4_code_to_f32(nibble) * sc);
            // No code leaving [-8, 7] and no NaN from a finite source.
            expect(std::isfinite(dq[n * K + k]));
            expect(std::fabs(w4_code_to_f32(nibble)) <= 8.0f);
        }
    }
    // Zero group: scale 0 and all-zero codes, decode exactly 0.
    luisa::vector<float> zeros(K, 0.0f);
    luisa::vector<uint32_t> zw(K / 8u, 0xFFFFFFFFu);
    luisa::vector<float> zs(K / 128u, 1.0f);
    expect(pack_w4_matrix(zeros, K, 1u, cfg, zw, zs, nullptr).ok());
    expect(zs[0] == 0.0f);
    for (auto x : zw) { expect(x == 0u); }
    // Non-finite input is refused; nothing is written.
    luisa::vector<float> bad(K * N, 0.0f);
    bad[7u] = std::numeric_limits<float>::quiet_NaN();
    luisa::vector<uint32_t> bw((K * N) / 8u, 0x12345678u);
    luisa::vector<float> bs(N * 2u, 7.0f);
    expect(!pack_w4_matrix(bad, K, N, cfg, bw, bs, nullptr).ok());
    expect(bw[0] == 0x12345678u && bs[0] == 7.0f);
    bad[7u] = std::numeric_limits<float>::infinity();
    expect(!pack_w4_matrix(bad, K, N, cfg, bw, bs, nullptr).ok());
    // Span/config validation.
    luisa::vector<uint32_t> too_small(3u, 0u);
    expect(!pack_w4_matrix(w, K, N, cfg, too_small, scales, nullptr).ok());
    expect(!pack_w4_matrix(w, K, N, W4ClipSearchConfig{96u, 0.85f, 1.0f, 31u}, words, scales, nullptr).ok());
    expect(!dequantize_w4_matrix(words, scales, K + 8u, N, 128u, dq).ok());
}

void test_w4_dsl_parity() {
    // The offline packer uses common.h::pack_w4_matrix; the DSL layer exposes
    // dsl::host_pack_w4_matrix/host_decode_packed for its own tests and references.
    // They MUST agree bit for bit, otherwise an offline pack and an in-kernel
    // reference would describe different weights.
    constexpr uint64_t K = 384u;// three 128-groups
    constexpr uint64_t N = 3u;
    luisa::vector<float> w(K * N);
    uint64_t state = 0x9E3779B97F4A7C15ull;
    for (auto &v : w) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const double u = static_cast<double>((state >> 11u) & 0xFFFFFFu) / 16777216.0;
        v = static_cast<float>((u - 0.5) * 0.25);
    }
    W4ClipSearchConfig cfg{};
    cfg.group = 128u;
    luisa::vector<uint32_t> cw((K * N) / 8u, 0u), dw((K * N) / 8u, 0u);
    luisa::vector<float> cs(N * 3u, 0.0f), ds(N * 3u, 0.0f);
    expect(pack_w4_matrix(w, K, N, cfg, cw, cs, nullptr).ok());
    expect(dsl::host_pack_w4_matrix(w, K, N, 128u, 0.85f, 1.0f, 31u, dw, ds, nullptr, nullptr, nullptr).ok());
    expect(std::memcmp(cw.data(), dw.data(), cw.size() * 4u) == 0);
    expect(std::memcmp(cs.data(), ds.data(), cs.size() * 4u) == 0);
    // Exact host/device decode parity: dsl::host_decode_packed == dequantize.
    luisa::vector<float> ref(K * N, 0.0f);
    expect(dequantize_w4_matrix(cw, cs, K, N, 128u, ref).ok());
    for (uint64_t i = 0; i < K * N; ++i) {
        expect(dsl::host_decode_packed(dsl::QwenWeightFormat::w4,
                                       luisa::span<const uint32_t>{cw.data(), cw.size()},
                                       luisa::span<const float>{cs.data(), cs.size()},
                                       K, 128u, i) == ref[i]);
    }
    // dsl weight-format geometry mirrors the host contract.
    expect(dsl::qwen_weight_pack(dsl::QwenWeightFormat::w4) == 8u);
    expect(dsl::qwen_weight_group(dsl::QwenWeightFormat::w4) == 128u);
    expect(dsl::validate_qwen_w4_geometry(2560u, 128u).ok());
    expect(!dsl::validate_qwen_w4_geometry(2540u, 128u).ok());
    auto pw = dsl::packed_matrix_words(dsl::QwenWeightFormat::w4, 2560u, 9216u);
    expect(pw.ok() && *pw.value == (2560u * 9216u) / 8u);
    auto se = dsl::weight_scale_elements(dsl::QwenWeightFormat::w4, 2560u, 9216u, 128u);
    expect(se.ok() && *se.value == 9216u * (2560u / 128u));
    expect(!dsl::weight_scale_elements(dsl::QwenWeightFormat::w4, 2560u, 8u, 96u).ok());
    // host_pack_matrix routes w4 through the canonical clipping search.
    luisa::vector<uint32_t> hw;
    luisa::vector<float> hs;
    expect(dsl::host_pack_matrix(dsl::QwenWeightFormat::w4, w, K, N, 128u, hw, hs).ok());
    expect(hw.size() == cw.size() && hs.size() == cs.size());
    expect(std::memcmp(hw.data(), cw.data(), cw.size() * 4u) == 0);
}

void test_w4_pack_manifest() {
    // Write a manifest into .kimix_cache and read it back: identical fields, and
    // a tampered field is refused. The payloads themselves are exercised by the
    // device upload test (upload_w4_pack) and by the offline packer run.
    const auto dir = luisa::filesystem::temp_directory_path() / "qwen_w4_manifest_test";
    std::error_code ec;
    luisa::filesystem::create_directories(dir / "tensors", ec);
    W4PackManifest m;
    m.version = kW4PackVersion;
    m.source_name = "fixture.gguf";
    m.source_size = 12345u;
    m.source_hash = 0x0123456789abcdefull;
    m.clip = W4ClipSearchConfig{128u, 0.85f, 1.0f, 31u};
    m.has_fp8_payloads = true;
    m.format_order = "w4,fp8";
    // Importance-aware provenance: the loader must survive (and report) it, and its
    // tensor_count must stay consistent with the per-kind counters (review A3).
    m.has_importance = true;
    m.importance_name = "fixture_importance";
    m.importance_hash = 0xfeedfacecafebeefull;
    m.importance_source_hash = 0xabcdef0123456789ull;
    m.importance_tokens = 128u;
    W4PackTensorRecord t;
    t.name = "blk.0.ffn_gate.weight";
    t.role = QwenTensorRole::ffn_gate;
    t.layer = 0u;
    t.k = 256u;
    t.n = 8u;
    t.storage = WeightStorage::w4_packed;
    t.group = 128u;
    t.source_bytes = 256u * 8u * 2u;
    t.source_hash = 0x1111ull;
    t.w4_bytes = (256u * 8u) / 8u * 4u;
    t.w4_hash = 0x2222ull;
    t.w4s_bytes = 8u * 2u * 4u;
    t.w4s_hash = 0x3333ull;
    t.fp8_bytes = (256u * 8u) / 4u * 4u;
    t.fp8_hash = 0x4444ull;
    t.fp8s_bytes = 8u * 4u * 4u;
    t.fp8s_hash = 0x5555ull;
    m.tensors.push_back(t);
    expect(write_w4_pack_manifest(dir, m).ok());
    auto back = read_w4_pack_manifest(dir);
    expect(back.ok());
    if (back.ok()) {
        const auto &b = *back.value;
        expect(b.version == kW4PackVersion);
        expect(b.source_name == "fixture.gguf");
        expect(b.source_size == 12345u);
        expect(b.source_hash == 0x0123456789abcdefull);
        expect(b.clip.group == 128u && b.clip.steps == 31u);
        expect(b.format_order == "w4,fp8");
        expect(b.tensors.size() == 1u);
        expect(b.tensors[0].name == t.name);
        expect(b.tensors[0].role == QwenTensorRole::ffn_gate);
        expect(b.tensors[0].layer == 0u);
        expect(b.tensors[0].w4_hash == 0x2222ull);
        expect(b.tensors[0].fp8s_hash == 0x5555ull);
    }
    // A manifest with a bad version is refused (no partial interpretation).
    {
        auto bad = m;
        bad.version = kW4PackVersion + 1u;
        expect(write_w4_pack_manifest(dir, bad).ok());
        auto r = read_w4_pack_manifest(dir);
        expect(!r.ok());
        expect(write_w4_pack_manifest(dir, m).ok());
    }
    // A ragged w4 geometry in the manifest is refused.
    {
        auto bad = m;
        bad.tensors[0].k = 2540u;
        // The w4 payload size stays consistent, so geometry validation is what fires.
        bad.tensors[0].w4_bytes = (2540u * 8u) / 8u * 4u;
        bad.tensors[0].w4s_bytes = 8u * ((2540u / 128u)) * 4u;
        expect(write_w4_pack_manifest(dir, bad).ok());
        auto r = read_w4_pack_manifest(dir);
        expect(!r.ok());
        expect(write_w4_pack_manifest(dir, m).ok());
    }
    std::filesystem::remove_all(dir, ec);
}

void test_device_upload() {
    const auto argc = g_argc;
    const auto argv = g_argv;
    if (argc <= 1 || argv == nullptr || argv[1] == nullptr || argv[1][0] == '\0') {
        return;// host-only run: skip device portion
    }
    TempDir tmp;
    const Qwen35Geometry g{};
    // Build the exact expected inventory (mirrors Qwen35Model::load's order).
    using T = std::tuple<luisa::string, uint32_t, luisa::vector<uint64_t>, luisa::vector<std::byte>>;
    luisa::vector<T> tensors;
    luisa::vector<std::pair<QwenTensorRole, uint64_t>> inventory;
    inventory.emplace_back(QwenTensorRole::token_embd, UINT64_MAX);
    inventory.emplace_back(QwenTensorRole::output_norm, UINT64_MAX);
    const QwenTensorRole shared[] = {
        QwenTensorRole::attn_norm, QwenTensorRole::post_attention_norm,
        QwenTensorRole::ffn_gate, QwenTensorRole::ffn_up, QwenTensorRole::ffn_down};
    const QwenTensorRole attention[] = {
        QwenTensorRole::attn_q, QwenTensorRole::attn_k, QwenTensorRole::attn_v,
        QwenTensorRole::attn_output, QwenTensorRole::attn_q_norm, QwenTensorRole::attn_k_norm};
    const QwenTensorRole gdn[] = {
        QwenTensorRole::gdn_qkv, QwenTensorRole::gdn_gate, QwenTensorRole::gdn_alpha,
        QwenTensorRole::gdn_beta, QwenTensorRole::gdn_out, QwenTensorRole::gdn_conv1d,
        QwenTensorRole::gdn_a, QwenTensorRole::gdn_dt_bias, QwenTensorRole::gdn_norm};
    for (uint64_t layer = 0; layer < g.layers; ++layer) {
        for (auto role : shared) { inventory.emplace_back(role, layer); }
        if (is_full_attention_layer(g, layer)) {
            for (auto role : attention) { inventory.emplace_back(role, layer); }
        } else {
            for (auto role : gdn) { inventory.emplace_back(role, layer); }
        }
    }
    expect(inventory.size() == 426u);
    // Frozen dims per role (ne order) + F32-required classification.
    auto dims_of = [&](QwenTensorRole role) -> luisa::vector<uint64_t> {
        using R = QwenTensorRole;
        switch (role) {
            case R::token_embd: return {g.hidden, g.vocab};
            case R::output_norm:
            case R::attn_norm:
            case R::post_attention_norm: return {g.hidden};
            case R::ffn_gate:
            case R::ffn_up: return {g.hidden, g.ffn};
            case R::ffn_down: return {g.ffn, g.hidden};
            case R::attn_q: return {g.hidden, g.q_proj_n};
            case R::attn_k:
            case R::attn_v: return {g.hidden, g.kv_proj_n};
            case R::attn_output: return {g.o_proj_k, g.hidden};
            case R::attn_q_norm:
            case R::attn_k_norm: return {g.head_dim};
            case R::gdn_qkv: return {g.hidden, g.gdn_qkv_n};
            case R::gdn_gate: return {g.hidden, g.gdn_z_n};
            case R::gdn_alpha:
            case R::gdn_beta: return {g.hidden, g.gdn_gate_n};
            case R::gdn_out: return {g.gdn_out_k, g.hidden};
            case R::gdn_conv1d: return {g.conv_kernel, g.gdn_qkv_n};
            case R::gdn_a:
            case R::gdn_dt_bias: return {g.gdn_v_heads};
            case R::gdn_norm: return {g.gdn_head_dim};
        }
        return {};
    };
    auto f32_required = [](QwenTensorRole role) {
        using R = QwenTensorRole;
        switch (role) {
            case R::output_norm:
            case R::attn_norm:
            case R::post_attention_norm:
            case R::attn_q_norm:
            case R::attn_k_norm:
            case R::gdn_conv1d:
            case R::gdn_a:
            case R::gdn_dt_bias:
            case R::gdn_norm: return true;
            default: return false;
        }
    };
    // Host source values per tensor (BF16 tensors hold the BF16-rounded value,
    // exactly what the GGUF payload encodes).
    luisa::vector<luisa::vector<float>> sources;
    for (size_t idx = 0; idx < inventory.size(); ++idx) {
        const auto [role, layer] = inventory[idx];
        const auto name = qwen_tensor_name(role, layer);
        const auto dims = dims_of(role);
        uint64_t count = 1;
        for (auto d : dims) { count *= d; }
        luisa::vector<float> vals(static_cast<size_t>(count));
        const bool bf16 = !f32_required(role);
        for (uint64_t i = 0; i < count; ++i) {
            float v = synth_value(static_cast<uint32_t>(idx), i);
            vals[static_cast<size_t>(i)] = bf16 ? bf16_to_f32(f32_to_bf16(v)) : v;
        }
        tensors.emplace_back(name, bf16 ? 30u : 0u, dims,
                             bf16 ? bf16_bytes(vals) : f32_bytes(vals));
        sources.emplace_back(std::move(vals));
    }
    const auto gguf = make_gguf(tensors, qwen35_meta());
    const auto file = tmp.path / "synthetic_qwen35.gguf";
    write_file(file, gguf);
    auto r = GgufReader::open(file);
    expect(r.ok());
    if (!r.ok()) { return; }
    auto model = Qwen35Model::load(**r.value);
    expect(model.ok());
    if (!model.ok()) {
        std::printf("synthetic model load failed: %s\n", model.status.message.c_str());
        return;
    }
    expect(model.value->tensors().size() == 426u);

    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    expect(static_cast<bool>(device));
    if (!device) { return; }
    auto stream = device.create_stream();

    // The REAL upload path, once per storage variant. chunk_bytes below the
    // 64 KiB clamp forces many streaming chunks per tensor.
    auto upload_and_check = [&](bool fp8) {
        QwenWeightUploadOptions options;
        options.fp8 = fp8;
        options.chunk_bytes = 1024u;// clamped to 64 KiB by the uploader
        QwenWeightUploadReport report;
        auto up = QwenDeviceWeights::upload(device, stream, **r.value, *model.value, options, &report);
        expect(up.ok());
        if (!up.ok()) {
            std::printf("upload failed (%s): %s\n", fp8 ? "fp8" : "bf16", up.status.message.c_str());
            return;
        }
        stream << luisa::compute::synchronize();
        const auto &weights = *up.value;
        expect(weights.tensors().size() == 426u);
        expect(report.tensor_count == 426u);
        // 2 globals (embd matrix + norm) + per-layer sensitive tensors:
        // shared 2 norms * 32 + attention 2 norms * 8 + gdn (conv,a,dt,norm) * 24.
        const uint64_t f32_count = 1u + 64u + 16u + 96u;
        expect(report.f32_count == f32_count);
        expect((fp8 ? report.fp8_count : report.bf16_count) == 426u - f32_count);
        expect((fp8 ? report.bf16_count : report.fp8_count) == 0u);
        if (fp8) {
            // The FP8 conversion is never silent: one fp8_quantized
            // diagnostic per converted matrix.
            uint64_t quant_diags = 0;
            for (const auto &d : report.diagnostics) {
                if (d.code == "fp8_quantized") { ++quant_diags; }
            }
            expect(quant_diags == 426u - f32_count);
        }
        // Round-trip EVERY tensor against the host source.
        for (size_t idx = 0; idx < inventory.size(); ++idx) {
            const auto [role, layer] = inventory[idx];
            const auto name = qwen_tensor_name(role, layer);
            const auto *t = weights.find(name);
            expect(t != nullptr);
            if (t == nullptr) { continue; }
            const auto &src = sources[idx];
            const uint64_t count = t->element_count;
            expect(count == src.size());
            const bool is_matrix = t->dims.size() == 2u;
            const WeightStorage want_storage =
                f32_required(role) ? WeightStorage::f32_dense
                : (fp8 && is_matrix ? WeightStorage::fp8_packed : WeightStorage::bf16_packed);
            expect(t->storage == want_storage);
            if (t->storage == WeightStorage::f32_dense) {
                luisa::vector<float> back(static_cast<size_t>(count));
                stream << t->f32.view().copy_to(luisa::span<float>{back});
                stream << luisa::compute::synchronize();
                bool ok = true;
                for (uint64_t i = 0; i < count && ok; ++i) { ok = back[i] == src[i]; }
                expect(ok);
                if (!ok) { std::printf("f32 mismatch in %s\n", name.c_str()); }
            } else if (t->storage == WeightStorage::bf16_packed) {
                const uint64_t words = (count + 1u) / 2u;
                luisa::vector<uint32_t> back(static_cast<size_t>(words));
                stream << t->packed.view().copy_to(luisa::span<uint32_t>{back});
                stream << luisa::compute::synchronize();
                bool ok = true;
                for (uint64_t i = 0; i < count && ok; ++i) {
                    const uint16_t code = static_cast<uint16_t>((back[i / 2u] >> (16u * (i % 2u))) & 0xffffu);
                    ok = bf16_to_f32(code) == src[i];
                }
                expect(ok);
                if (!ok) { std::printf("bf16 mismatch in %s\n", name.c_str()); }
            } else {
                const uint64_t k = t->dims[0], n = t->dims[1];
                const uint64_t words = (count + 3u) / 4u;
                const uint64_t gpr = fp8_group_count(k);
                expect(t->groups_per_row == gpr);
                luisa::vector<uint32_t> back(static_cast<size_t>(words));
                luisa::vector<float> bscales(static_cast<size_t>(n * gpr));
                stream << t->packed.view().copy_to(luisa::span<uint32_t>{back});
                stream << t->scales.view().copy_to(luisa::span<float>{bscales});
                stream << luisa::compute::synchronize();
                // Host reference quantization of the SAME source values.
                luisa::vector<uint32_t> host(static_cast<size_t>(words));
                luisa::vector<float> hscales(static_cast<size_t>(n * gpr));
                expect(quantize_fp8_matrix(src, k, n, host, hscales).ok());
                bool ok = true;
                for (uint64_t i = 0; i < words && ok; ++i) { ok = back[i] == host[i]; }
                for (uint64_t i = 0; i < n * gpr && ok; ++i) { ok = bscales[i] == hscales[i]; }
                expect(ok);
                if (!ok) { std::printf("fp8 mismatch in %s\n", name.c_str()); }
                // Dequantize error bound (E4M3: 3 mantissa bits).
                luisa::vector<float> dq(static_cast<size_t>(count));
                expect(dequantize_fp8_matrix(back, bscales, k, n, dq).ok());
                for (uint64_t i = 0; i < count; ++i) {
                    expect(std::abs(dq[i] - src[i]) <= std::abs(src[i]) * 0.0625f + 0.002f);
                }
            }
        }
    };
    upload_and_check(false);
    upload_and_check(true);
 // -----------------------------------------------------------------------
 // OFFLINE merged projection pack, end to end (opt_plan/03 section 3 + D5).
 //
 // Proves, on the SAME synthetic checkpoint:
 //  * the offline packer writes the canonical inventory (32 matrices / 120
 //    segments) and its payloads + rolling source digest verify;
 //  * a pack whose recorded source identity differs is REFUSED (size and
 //    digest are both checked);
 //  * uploading WITH the pack SKIPS the 120 source tensors and creates the 32
 //    merged matrices plus their 120 segment aliases -- device_bytes is
 //    UNCHANGED, i.e. the pack replaces the segments instead of duplicating them;
 //  * `find(role, layer)` still returns the per-role tensor (now an alias whose
 //    packed view is the segment's row range) and merged_parent_of() walks back;
 //  * the merged payload is BYTE-EXACT the row concatenation of the sources, in
 //    canonical segment order (sampled rows per segment), which is what makes the
 //    merged decode GEMV numerically identical to the unmerged one;
 //  * a pack of the wrong FORMAT is refused instead of being re-quantized.
 // -----------------------------------------------------------------------
 {
     ProjectionPackOptions pack_options;
     pack_options.format = WeightStorage::bf16_packed;
     pack_options.group = 0u;
     pack_options.chunk_bytes = 64ull * 1024ull;
     const auto pack_dir = tmp.path / "proj-pack-bf16";
     ProjectionPackReport pack_report;
     expect(pack_projection_pack_from_gguf(**r.value, *model.value, "synthetic_qwen35.gguf",
                                           pack_dir, pack_options, &pack_report).ok());
     expect(pack_report.matrices == 32u);
     expect(pack_report.segments == 120u);
     expect(pack_report.word_bytes != 0u);
     auto pack_manifest = read_projection_pack_manifest(pack_dir);
     expect(pack_manifest.ok());
     auto pack_inventory = qwen35_merged_projection_inventory(*model.value);
     expect(pack_inventory.ok());
     if (!pack_manifest.ok() || !pack_inventory.ok()) { return; }
     expect(validate_projection_pack_manifest(*pack_manifest.value, *pack_inventory.value).ok());
     expect(verify_projection_pack_payloads(pack_dir, *pack_manifest.value).ok());
     // The digest the packer recorded is exactly the one this checkpoint reproduces.
     auto digest = projection_pack_source_digest(**r.value, *model.value);
     expect(digest.ok());
     if (digest.ok()) { expect(*digest.value == pack_manifest.value->source_hash); }
     expect(verify_projection_pack_source(**r.value, *model.value, *pack_manifest.value).ok());
     {
         // Same-size / different-weights identity: tamper the recorded digest.
         auto bad = *pack_manifest.value;
         bad.source_hash ^= 0xFFFFull;
         expect(!verify_projection_pack_source(**r.value, *model.value, bad).ok());
         // A same-file / different-SIZE checkpoint is refused by the size gate.
         auto bad_size = *pack_manifest.value;
         bad_size.source_size += 32u;
         expect(!verify_projection_pack_source(**r.value, *model.value, bad_size).ok());
     }

     // Unmerged reference upload of the same variant: the device byte count the
     // merged upload must match (no duplicated residency).
     QwenWeightUploadOptions plain_options;
     plain_options.chunk_bytes = 64ull * 1024ull;
     QwenWeightUploadReport plain_report;
     auto plain_up = QwenDeviceWeights::upload(device, stream, **r.value, *model.value,
                                               plain_options, &plain_report);
     expect(plain_up.ok());
     stream << luisa::compute::synchronize();

     QwenWeightUploadOptions merged_options = plain_options;
     QwenProjectionPackSource pack_source{pack_dir, &*pack_manifest.value};
     merged_options.projection_pack = &pack_source;
     QwenWeightUploadReport merged_report;
     auto merged_up = QwenDeviceWeights::upload(device, stream, **r.value, *model.value,
                                                merged_options, &merged_report);
     expect(merged_up.ok());
     if (!merged_up.ok() || !plain_up.ok()) {
         std::printf("merged upload failed: %s (%s)\n",
                     merged_up.ok() ? plain_up.status.message.c_str()
                                    : merged_up.status.message.c_str(),
                     merged_up.ok() ? plain_up.status.detail.c_str()
                                    : merged_up.status.detail.c_str());
         return;
     }
     stream << luisa::compute::synchronize();
     expect(merged_report.projection_pack_skipped_tensors == 120u);
     expect(merged_report.projection_pack_matrices == 32u);
     expect(merged_report.projection_pack_segments == 120u);
     expect(merged_report.projection_pack_format == "bf16_packed");
     // The 120 source tensors are NOT uploaded; the pack adds 32 merged matrices
// plus 120 ALIASES that are sub-views of those matrices (no separate buffers).
expect(merged_report.tensor_count == 426u - 120u);
     // THE replacement proof: same device bytes as the unmerged upload.
     expect(merged_report.device_bytes == plain_report.device_bytes);

     const auto &mw = *merged_up.value;
     expect(mw.merged_matrices().size() == 32u);
expect(mw.tensors().size() == 426u - 120u + 32u + 120u);
     expect(mw.projection_pack_source() == "synthetic_qwen35.gguf");
     expect(mw.projection_pack_format() == "bf16_packed");
     // Every merged role resolves to an alias whose view is exactly its row range,
     // and the alias walks back to a parent with the canonical descriptor.
     const auto role_index = [&](QwenTensorRole role, uint64_t layer) -> size_t {
         for (size_t i = 0; i < inventory.size(); ++i) {
             if (inventory[i].first == role && inventory[i].second == layer) { return i; }
         }
         return SIZE_MAX;
     };
     const auto check_alias = [&](QwenTensorRole role, uint64_t layer) {
         const auto *alias = mw.find(role, layer);
         expect(alias != nullptr);
         if (alias == nullptr) { return; }
         expect(alias->merge_parent_index != UINT64_MAX);
         const auto *parent = mw.merged_parent_of(*alias);
         expect(parent != nullptr);
         if (parent == nullptr) { return; }
         expect(parent->merge_kind != QwenMergeKind::none);
         expect(alias->packed_view.size() ==
                alias->dims[1] * (alias->dims[0] / 2u));
         bool found = false;
         for (const auto &seg : parent->segments) {
             if (seg.role == role) {
                 found = true;
                 expect(seg.rows == alias->dims[1]);
                 expect(seg.row_offset == alias->merge_row_offset);
             }
         }
         expect(found);
     };
     for (uint64_t l = 0; l < g.layers; l++) {
         if (is_full_attention_layer(g, l)) {
             check_alias(QwenTensorRole::attn_q, l);
             check_alias(QwenTensorRole::attn_k, l);
             check_alias(QwenTensorRole::attn_v, l);
         } else {
             check_alias(QwenTensorRole::gdn_qkv, l);
             check_alias(QwenTensorRole::gdn_gate, l);
             check_alias(QwenTensorRole::gdn_alpha, l);
             check_alias(QwenTensorRole::gdn_beta, l);
         }
     }
     // Byte-exact merge: read the first two rows and the last row of every
     // segment and compare them with the host pack of the SOURCE payload.
     const auto read_words = [&](const compute::BufferView<uint32_t> &view, size_t count) {
         luisa::vector<uint32_t> out(count);
         stream << view.copy_to(luisa::span<uint32_t>{out.data(), out.size()});
         stream.synchronize();
         return out;
     };
     bool concat_ok = true;
     for (size_t mi = 0; mi < pack_manifest.value->matrices.size() && concat_ok; ++mi) {
         const auto &rec = pack_manifest.value->matrices[mi];
         const auto k = rec.k;
         const auto words_per_row = k / 2u;
         for (const auto &seg : rec.segments) {
             const auto *alias = mw.find(seg.role, rec.layer);
             if (alias == nullptr) { concat_ok = false; break; }
             const auto idx = role_index(seg.role, rec.layer);
             if (idx == SIZE_MAX) { concat_ok = false; break; }
             const auto &src = sources[idx];
             const uint64_t rows[] = {0u, 1u, seg.rows - 1u};
             for (uint32_t r = 0; r < 3u; r++) {
                 const uint64_t row = rows[r];
                 auto got = read_words(alias->packed_view.subview(
                                           static_cast<size_t>(row * words_per_row),
                                           static_cast<size_t>(words_per_row)),
                                       static_cast<size_t>(words_per_row));
                 for (uint64_t w = 0; w < words_per_row; w++) {
                     const uint16_t lo = f32_to_bf16(src[static_cast<size_t>(row * k + 2u * w)]);
                     const uint16_t hi = f32_to_bf16(src[static_cast<size_t>(row * k + 2u * w + 1u)]);
                     const uint32_t want = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16u);
                     if (got[static_cast<size_t>(w)] != want) {
                         concat_ok = false;
                         std::printf("merged byte-exact mismatch: matrix %s segment %s row %llu word %llu\n",
                                     rec.name.c_str(), seg.name.c_str(),
                                     static_cast<unsigned long long>(row),
                                     static_cast<unsigned long long>(w));
                         break;
                     }
                 }
                 if (!concat_ok) { break; }
             }
             if (!concat_ok) { break; }
         }
     }
     expect(concat_ok);

     // Format gate: an FP8 pack cannot serve a BF16 session (no re-quantization).
     auto fp8_options = pack_options;
     fp8_options.format = WeightStorage::fp8_packed;
     fp8_options.group = kFp8GroupSize;
     const auto fp8_pack_dir = tmp.path / "proj-pack-fp8";
     expect(pack_projection_pack_from_gguf(**r.value, *model.value, "synthetic_qwen35.gguf",
                                           fp8_pack_dir, fp8_options, nullptr).ok());
     auto fp8_manifest = read_projection_pack_manifest(fp8_pack_dir);
     expect(fp8_manifest.ok());
     if (fp8_manifest.ok()) {
         QwenProjectionPackSource fp8_source{fp8_pack_dir, &*fp8_manifest.value};
         auto wrong = plain_options;// bf16 upload, fp8 pack
         wrong.projection_pack = &fp8_source;
         QwenWeightUploadReport wrong_report;
         auto refused = QwenDeviceWeights::upload(device, stream, **r.value, *model.value,
                                                  wrong, &wrong_report);
         expect(!refused.ok());
         // ... and the mirrored mismatch (fp8 upload, bf16 pack) is refused too.
         auto wrong2 = plain_options;
         wrong2.fp8 = true;
         wrong2.projection_pack = &pack_source;
         auto refused2 = QwenDeviceWeights::upload(device, stream, **r.value, *model.value,
                                                   wrong2, &wrong_report);
         expect(!refused2.ok());
     }
 }
}


// ---------------------------------------------------------------------------
// Native Q8_0 DEVICE UPLOAD of the real checkpoint (phase 8a).
//
// Runs the real QwenDeviceWeights::upload against data/Qwen3.5-4B-Q8_0.gguf
// with the q8_0 opt-in and proves on the DEVICE side that the format is a byte
// copy of the checkpoint payload: the packed words and the FP32 scales are
// compared against the GGUF block bytes for bounded samples of several real
// matrices (token_embd, ffn_gate, gdn_qkv, ffn_down), and the decoded values
// against int8(code) * scale. Skipped when the checkpoint is absent.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Offline W4 pack round trip on the DEVICE (opt_plan/05 Q2 + section 5).
//
// A tiny two-tensor pack is written by the test itself (one W4 matrix with its
// FP8 partner payload, one dense F32 vector), loaded through upload_w4_pack and
// compared element by element against the host quantization. Then ONE payload
// byte is flipped and the loader must refuse the pack (hash fail-closed), and a
// manifest whose source size was changed must be refused by the CLI-level check
// (covered in the packer smoke; here the manifest version/geometry refusals are
// asserted). Host-only parts run in every invocation; the device part needs a
// backend argument like test_device_upload.
// ---------------------------------------------------------------------------
void test_w4_pack_upload() {
    constexpr uint64_t K = 256u;  // two 128-groups
    constexpr uint64_t N = 3u;    // small matrix
    const auto dir = luisa::filesystem::temp_directory_path() / "qwen_w4_pack_upload_test";
    std::error_code ec;
    luisa::filesystem::remove_all(dir, ec);
    luisa::filesystem::create_directories(dir / "tensors", ec);
    // Source values: deterministic with per-group outliers.
    luisa::vector<float> w(K * N);
    for (uint64_t n = 0; n < N; ++n) {
        for (uint64_t k = 0; k < K; ++k) {
            const float base = (k / 128u) == 0u ? 0.05f : 0.02f;
            w[n * K + k] = std::sin(0.31f * static_cast<float>(k) + static_cast<float>(n)) * base;
            if (k % 128u == 7u) { w[n * K + k] *= 5.0f; }
        }
    }
    W4ClipSearchConfig clip{};
    // pack_w4_matrix writes through caller-sized spans.
    luisa::vector<uint32_t> w4_words((K * N) / 8u, 0u);
    luisa::vector<float> w4_scales(N * (K / 128u), 0.0f);
    expect(pack_w4_matrix(w, K, N, clip,
                          luisa::span<uint32_t>{w4_words.data(), w4_words.size()},
                          luisa::span<float>{w4_scales.data(), w4_scales.size()}, nullptr).ok());
    luisa::vector<uint32_t> fp8_words((K * N + 3u) / 4u, 0u);
    luisa::vector<float> fp8_scales(N * (K / kFp8GroupSize), 0.0f);
    expect(quantize_fp8_matrix(w, K, N, fp8_words, fp8_scales).ok());
    luisa::vector<float> dense{1.0f, -2.5f, 0.25f};
    const auto write_bytes = [&](const luisa::filesystem::path &p, const void *data, uint64_t bytes) {
        std::ofstream out(p, std::ios::binary | std::ios::trunc);
        expect(out.is_open());
        out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
        expect(static_cast<bool>(out));
    };
    const auto w4_path = dir / "tensors" / "0.w4.bin";
    const auto w4s_path = dir / "tensors" / "0.w4s.bin";
    const auto fp8_path = dir / "tensors" / "0.fp8.bin";
    const auto fp8s_path = dir / "tensors" / "0.fp8s.bin";
    const auto f32_path = dir / "tensors" / "1.f32.bin";
    write_bytes(w4_path, w4_words.data(), w4_words.size() * 4u);
    write_bytes(w4s_path, w4_scales.data(), w4_scales.size() * 4u);
    write_bytes(fp8_path, fp8_words.data(), fp8_words.size() * 4u);
    write_bytes(fp8s_path, fp8_scales.data(), fp8_scales.size() * 4u);
    write_bytes(f32_path, dense.data(), dense.size() * 4u);
    const auto hash_of = [](const auto &vec) {
        return qwen_fnv1a64(luisa::span<const std::byte>{
            reinterpret_cast<const std::byte *>(vec.data()), vec.size() * sizeof(vec[0])});
    };
    W4PackManifest m;
    m.version = kW4PackVersion;
    m.source_name = "fixture.gguf";
    m.source_size = 4096u;
    m.source_hash = 0xabcdef0123456789ull;
    m.clip = clip;
    m.has_fp8_payloads = true;
    m.format_order = "w4,fp8";
    W4PackTensorRecord mat;
    mat.name = "blk.0.ffn_gate.weight";
    mat.role = QwenTensorRole::ffn_gate;
    mat.layer = 0u;
    mat.k = K;
    mat.n = N;
    mat.storage = WeightStorage::w4_packed;
    mat.group = clip.group;
    mat.source_bytes = K * N * 2u;
    mat.source_hash = 0x1111ull;
    mat.w4_bytes = w4_words.size() * 4u;
    mat.w4_hash = hash_of(w4_words);
    mat.w4s_bytes = w4_scales.size() * 4u;
    mat.w4s_hash = hash_of(w4_scales);
    mat.fp8_bytes = fp8_words.size() * 4u;
    mat.fp8_hash = hash_of(fp8_words);
    mat.fp8s_bytes = fp8_scales.size() * 4u;
    mat.fp8s_hash = hash_of(fp8_scales);
    W4PackTensorRecord dense_rec;
    dense_rec.name = "blk.0.attn_norm.weight";
    dense_rec.role = QwenTensorRole::attn_norm;
    dense_rec.layer = 0u;
    dense_rec.k = dense.size();
    dense_rec.n = 1u;
    dense_rec.storage = WeightStorage::f32_dense;
    dense_rec.source_bytes = dense.size() * 4u;
    dense_rec.source_hash = 0x2222ull;
    dense_rec.f32_bytes = dense.size() * 4u;
    dense_rec.f32_hash = hash_of(dense);
    m.tensors.push_back(mat);
    m.tensors.push_back(dense_rec);
    expect(write_w4_pack_manifest(dir, m).ok());
    // Round trip of the manifest itself.
    auto back = read_w4_pack_manifest(dir);
    expect(back.ok());
    if (back.ok()) {
        expect(back.value->tensors.size() == 2u);
        expect(back.value->tensors[0].w4_hash == mat.w4_hash);
        expect(back.value->tensors[1].storage == WeightStorage::f32_dense);
    }
    // Wrong version / ragged geometry are refused by the reader.
    {
        auto bad = m;
        bad.version = 99u;
        expect(write_w4_pack_manifest(dir, bad).ok());
        expect(!read_w4_pack_manifest(dir).ok());
        expect(write_w4_pack_manifest(dir, m).ok());
    }
    const auto argc = g_argc;
    const auto argv = g_argv;
    if (argc <= 1 || argv == nullptr || argv[1] == nullptr || argv[1][0] == '\0') {
        std::filesystem::remove_all(dir, ec);
        return;// host-only run: the pack writer/manifest half already ran
    }
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    expect(static_cast<bool>(device));
    if (!device) {
        std::filesystem::remove_all(dir, ec);
        return;
    }
    auto stream = device.create_stream();
    // 1) The pack loads, exposes the decode tensor AND its "<name>.fp8" partner,
    //    and the values are the host ABI values exactly.
    {
        QwenWeightUploadReport report;
        auto up = upload_w4_pack(device, stream, dir, 64ull * 1024ull, &report);
        expect(up.ok());
        if (up.ok()) {
            expect(report.w4_count == 1u);
            expect(report.fp8_count == 1u);
            expect(report.f32_count == 1u);
            // Review A3: two manifest RECORDS produce THREE device tensors, so tensor_count
            // must count device tensors and stay consistent with the per-kind counters (the
            // pre-fix code reported the record count).
            expect(report.manifest_records == 2u);
            expect(report.tensor_count == 3u);
            expect(report.tensor_count == report.w4_count + report.fp8_count + report.f32_count +
                                        report.bf16_count + report.q8_0_count);
            {
             bool saw_importance = false;
             for (const auto &d : report.diagnostics) {
             saw_importance = saw_importance || d.code == "w4_pack_importance_aware";
             }
             expect(saw_importance);
            }
            const auto &weights = *up.value;
            expect(weights.tensors().size() == 3u);
            const auto *t = weights.find("blk.0.ffn_gate.weight");
            expect(t != nullptr);
            expect(weights.find("blk.0.ffn_gate.weight.fp8") != nullptr);
            if (t != nullptr) {
                expect(t->storage == WeightStorage::w4_packed);
                expect(t->group == 128u);
                expect(t->groups_per_row == K / 128u);
                luisa::vector<uint32_t> words(t->packed.size());
                luisa::vector<float> scales(t->scales.size());
                stream << t->packed.view().copy_to(luisa::span<uint32_t>{words});
                stream << t->scales.view().copy_to(luisa::span<float>{scales});
                stream << luisa::compute::synchronize();
                expect(words.size() == w4_words.size());
                expect(std::memcmp(words.data(), w4_words.data(), words.size() * 4u) == 0);
                expect(std::memcmp(scales.data(), w4_scales.data(), scales.size() * 4u) == 0);
                // Decode parity for a sample of elements.
                luisa::vector<float> ref(K * N);
                expect(dequantize_w4_matrix(w4_words, w4_scales, K, N, 128u, ref).ok());
                bool parity = true;
                for (uint64_t i = 0; i < K * N; i += 17u) {
                    const uint32_t nibble = (words[i / 8u] >> (4u * (i % 8u))) & 0xFu;
                    parity = parity && (ref[i] == w4_code_to_f32(nibble) * scales[(i / K) * 2u + (i % K) / 128u]);
                }
                expect(parity);
            }
            const auto *d = weights.find("blk.0.attn_norm.weight");
            expect(d != nullptr && d->storage == WeightStorage::f32_dense);
            if (d != nullptr) {
                luisa::vector<float> got(dense.size());
                stream << d->f32.view().copy_to(luisa::span<float>{got}) << luisa::compute::synchronize();
                expect(std::memcmp(got.data(), dense.data(), dense.size() * 4u) == 0);
            }
        }
    }
    // 2) One flipped payload byte must fail the load (hash fail-closed).
    {
        std::fstream f(w4_path, std::ios::binary | std::ios::in | std::ios::out);
        expect(f.is_open());
        char byte = 0;
        f.seekg(0);
        f.read(&byte, 1);
        byte = static_cast<char>(byte ^ 0x01);
        f.seekp(0);
        f.write(&byte, 1);
        f.close();
        QwenWeightUploadReport report;
        auto up = upload_w4_pack(device, stream, dir, 64ull * 1024ull, &report);
        expect(!up.ok());
        if (!up.ok()) { expect(up.status.code == StatusCode::unvalidated); }
        // Restore the payload for a clean teardown path.
        write_bytes(w4_path, w4_words.data(), w4_words.size() * 4u);
    }
    // 3) A truncated payload is refused too (size check, before the hash).
    {
        std::error_code ec2;
        const auto bytes = luisa::filesystem::file_size(w4s_path, ec2);
        luisa::filesystem::resize_file(w4s_path, bytes - 4u, ec2);
        auto up = upload_w4_pack(device, stream, dir, 64ull * 1024ull, nullptr);
        expect(!up.ok());
        write_bytes(w4s_path, w4_scales.data(), w4_scales.size() * 4u);
    }
    std::filesystem::remove_all(dir, ec);
}

void test_device_upload_q8_real() {
    const auto argc = g_argc;
    const auto argv = g_argv;
    if (argc <= 1 || argv == nullptr || argv[1] == nullptr || argv[1][0] == '\0') {
        return;// host-only run
    }
#ifndef LUISA_QWEN_TEST_DATA_DIR
#define LUISA_QWEN_TEST_DATA_DIR "data"
#endif
    const luisa::filesystem::path file{LUISA_QWEN_TEST_DATA_DIR "/Qwen3.5-4B-Q8_0.gguf"};
    std::error_code ec;
    if (!luisa::filesystem::exists(file, ec)) { return; }
    auto r = GgufReader::open(file);
    expect(r.ok());
    if (!r.ok()) { return; }
    auto model = Qwen35Model::load(**r.value);
    expect(model.ok());
    if (!model.ok()) { return; }
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    expect(static_cast<bool>(device));
    if (!device) { return; }
    auto stream = device.create_stream();
    // Without the opt-in the real checkpoint is refused before any allocation
    // succeeds (first Q8_0 tensor), i.e. the default stays fail-closed.
    {
        QwenWeightUploadOptions off;
        auto refused = QwenDeviceWeights::upload(device, stream, **r.value, *model.value, off);
        expect(!refused.ok());
        expect(refused.status.code == StatusCode::unsupported);
        expect(refused.status.message.find("Q8_0") != luisa::string::npos);
    }
    QwenWeightUploadOptions options;
    options.q8_0 = true;
    QwenWeightUploadReport report;
    auto up = QwenDeviceWeights::upload(device, stream, **r.value, *model.value, options, &report);
    expect(up.ok());
    if (!up.ok()) {
        std::printf("real q8 upload failed: %s\n", up.status.message.c_str());
        return;
    }
    stream << luisa::compute::synchronize();
    const auto &weights = *up.value;
    expect(weights.tensors().size() == 426u);
    expect(report.tensor_count == 426u);
    // The real checkpoint stores 177 F32 tensors (norms/conv/a/dt) and 249 Q8_0
    // matrices; nothing is converted.
    expect(report.f32_count == 177u);
    expect(report.q8_0_count == 249u);
    expect(report.bf16_count == 0u && report.fp8_count == 0u);
    uint64_t native_diags = 0, quant_diags = 0;
    for (const auto &d : report.diagnostics) {
        if (d.code == "q8_0_native") { ++native_diags; }
        if (d.code == "fp8_quantized") { ++quant_diags; }
    }
    expect(native_diags == 1u);
    expect(quant_diags == 0u);
    // Bounded device-vs-checkpoint comparison on real matrices: the whole first
    // 1 MiB of packed words plus a strided sample across the rest, and the
    // scales that belong to those blocks.
    const auto check_tensor = [&](const char *name) {
        const auto *t = weights.find(name);
        expect(t != nullptr);
        if (t == nullptr) { return; }
        const auto *info = (*r.value)->find(name);
        expect(info != nullptr);
        if (info == nullptr) { return; }
        expect(t->storage == WeightStorage::q8_0_packed);
        if (t->storage != WeightStorage::q8_0_packed) { return; }
        const uint64_t k = t->dims[0];
        expect(t->groups_per_row == k / kQ8_0BlockSize);
        const uint64_t words = t->packed.size();
        expect(words == t->element_count / 4u);
        const uint64_t sample_words = std::min<uint64_t>(words, 1ull << 18u);// 1 MiB
        // First `sample_words` words: read the matching GGUF block bytes and
        // compare every byte, then every scale.
        luisa::vector<uint32_t> dev(sample_words);
        stream << t->packed.view(0, static_cast<size_t>(sample_words))
                      .copy_to(luisa::span<uint32_t>{dev});
        stream << luisa::compute::synchronize();
        const uint64_t blocks = sample_words / (kQ8_0BlockSize / 4u);
        luisa::vector<std::byte> raw(blocks * kQ8_0BlockBytes);
        expect((*r.value)->read_tensor_bytes_at(*info, 0u, raw).ok());
        bool bytes_ok = true;
        for (uint64_t b = 0; b < blocks && bytes_ok; ++b) {
            for (uint64_t i = 0; i < kQ8_0BlockSize; i++) {
                const uint32_t want = static_cast<uint32_t>(
                    static_cast<uint8_t>(raw[static_cast<size_t>(b * kQ8_0BlockBytes + 2u + i)]));
                const uint32_t got =
                    (dev[static_cast<size_t>(b * 8u + i / 4u)] >> (8u * (i % 4u))) & 0xFFu;
                if (got != want) { bytes_ok = false; break; }
            }
        }
        expect(bytes_ok);
        // Scales: exact FP16 -> FP32 widening of the same blocks.
        luisa::vector<float> sc(blocks);
        stream << t->scales.view(0, static_cast<size_t>(blocks))
                      .copy_to(luisa::span<float>{sc});
        stream << luisa::compute::synchronize();
        bool scales_ok = true;
        for (uint64_t b = 0; b < blocks && scales_ok; ++b) {
            uint16_t bits;
            std::memcpy(&bits, raw.data() + b * kQ8_0BlockBytes, 2u);
            scales_ok = sc[static_cast<size_t>(b)] == fp16_to_f32(bits);
        }
        expect(scales_ok);
        // Decoded weights equal int8(code) * scale exactly (no rounding at all).
        bool decode_ok = true;
        for (uint64_t b = 0; b < blocks && decode_ok; ++b) {
            float scale;
            uint16_t bits;
            std::memcpy(&bits, raw.data() + b * kQ8_0BlockBytes, 2u);
            scale = fp16_to_f32(bits);
            for (uint64_t i = 0; i < kQ8_0BlockSize; i++) {
                const auto code = static_cast<uint8_t>(
                    raw[static_cast<size_t>(b * kQ8_0BlockBytes + 2u + i)]);
                const float want = static_cast<float>(static_cast<int8_t>(code)) * scale;
                const uint64_t idx = b * kQ8_0BlockSize + i;
                const uint8_t got_code = static_cast<uint8_t>(
                    (dev[static_cast<size_t>(idx / 4u)] >> (8u * (idx % 4u))) & 0xFFu);
                if (static_cast<float>(static_cast<int8_t>(got_code)) * sc[static_cast<size_t>(b)] != want) {
                    decode_ok = false;
                    break;
                }
            }
        }
        expect(decode_ok);
    };
    check_tensor("token_embd.weight");
    check_tensor("blk.0.ffn_gate.weight");
    check_tensor("blk.0.ffn_down.weight");
    check_tensor("blk.0.attn_qkv.weight");// the GDN packed QKV projection
    check_tensor("blk.3.attn_q.weight");   // a full-attention Q projection
    // One dense F32 tensor stays bit-exact (no precision change for norms).
    const auto *norm = weights.find("output_norm.weight");
    expect(norm != nullptr && norm->storage == WeightStorage::f32_dense);
    if (norm != nullptr) {
        luisa::vector<float> dev(norm->element_count);
        stream << norm->f32.view().copy_to(luisa::span<float>{dev});
        stream << luisa::compute::synchronize();
        luisa::vector<float> ref(norm->element_count);
        expect((*r.value)->read_tensor_f32(*(*r.value)->find("output_norm.weight"), ref).ok());
        bool ok = true;
        for (uint64_t i = 0; i < norm->element_count && ok; ++i) { ok = dev[i] == ref[i]; }
        expect(ok);
    }
}

// ---------------------------------------------------------------------------
// Phase 8b review -- W4 source identity, aggregate report statistics, device
// tensor counts and importance-aware (activation-second-moment) clipping.
//
// A1: the manifest's source hash is a ROLLING PAYLOAD DIGEST over the inventory;
//     these tests recompute it from the raw file bytes and prove that a
//     SAME-SIZE / different-payload checkpoint is refused (a size check cannot).
// A2: W4PackReport.mse/rtn_mse are element-weighted means and mean_ratio is
//     group-weighted across the whole pack, computed once at the end.
// A3: upload_w4_pack's tensor_count counts DEVICE tensors (one W4 record can
//     create a W4 tensor AND its .fp8 partner).
// B : the weighted clipping objective changes the chosen scale in the expected
//     direction, uniform weights reproduce the clipping-only search byte for
//     byte, the importance file is checked, and the packer consumes it.
// ---------------------------------------------------------------------------

// Deterministic tiny checkpoint: 2 W4 matrices (K = 256 / 128, group 128) and 2
// dense tensors. The payload depends on `salt`, so two builds have the SAME byte
// size and different contents -- exactly the stale-checkpoint case.
luisa::filesystem::path write_tiny_w4_checkpoint(const luisa::filesystem::path &dir, uint32_t salt) {
    std::error_code ec;
    luisa::filesystem::create_directories(dir, ec);
    struct Spec {
        const char *name;
        uint32_t type;// 0 = F32, 30 = BF16
        luisa::vector<uint64_t> dims;
    };
    const Spec specs[] = {
        {"blk.0.ffn_gate.weight", 30u, {256u, 3u}},
        {"blk.0.ffn_down.weight", 0u, {128u, 2u}},
        {"blk.0.attn_norm.weight", 0u, {16u}},
        {"blk.0.ssm_conv1d.weight", 0u, {4u, 8u}},
    };
    luisa::vector<std::tuple<luisa::string, uint32_t, luisa::vector<uint64_t>, luisa::vector<std::byte>>> tensors;
    for (const auto &spec : specs) {
        uint64_t count = 1;
        for (auto d : spec.dims) { count *= d; }
        luisa::vector<float> vals(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            const float base = std::sin(static_cast<float>(i % 397u) * 0.031f +
                                        static_cast<float>(salt) * 0.7f) *
                               0.2f;
            // One outlier per 128-group so the clipping search has work to do.
            vals[static_cast<size_t>(i)] = (i % 128u == 7u) ? 5.0f * base : base;
        }
        tensors.emplace_back(luisa::string{spec.name}, spec.type, spec.dims,
                             spec.type == 30u ? bf16_bytes(vals) : f32_bytes(vals));
    }
    const auto bytes = make_gguf(tensors, {});
    const auto path = dir / luisa::filesystem::path{
                                 std::string{"tiny_"} + std::to_string(salt) + ".gguf"};
    write_file(path, bytes);
    return path;
}

luisa::vector<QwenTensorRef> tiny_w4_inventory(const GgufReader &reader) {
    const std::pair<const char *, QwenTensorRole> order[] = {
        {"blk.0.ffn_gate.weight", QwenTensorRole::ffn_gate},
        {"blk.0.ffn_down.weight", QwenTensorRole::ffn_down},
        {"blk.0.attn_norm.weight", QwenTensorRole::attn_norm},
        {"blk.0.ssm_conv1d.weight", QwenTensorRole::gdn_conv1d},
    };
    luisa::vector<QwenTensorRef> inventory;
    for (const auto &[name, role] : order) {
        const auto *info = reader.find(name);
        expect(info != nullptr);
        if (info != nullptr) { inventory.push_back(QwenTensorRef{role, 0u, info}); }
    }
    return inventory;
}

luisa::span<const QwenTensorRef> tiny_span(const luisa::vector<QwenTensorRef> &inventory) {
    return luisa::span<const QwenTensorRef>{inventory.data(), inventory.size()};
}

// Independent recomputation of ONE tensor's raw payload hash straight from the
// file bytes, in bounded chunks: the per-tensor quantity the manifest records.
uint64_t file_payload_hash(const luisa::filesystem::path &path, const GgufTensorInfo &info,
                           uint64_t chunk) {
    std::ifstream in{path, std::ios::binary};
    expect(static_cast<bool>(in));
    in.seekg(static_cast<std::streamoff>(info.data_offset));
    luisa::vector<std::byte> buf(static_cast<size_t>(std::max<uint64_t>(1u, chunk)));
    uint64_t hash = kFnv1aOffsetBasis;
    uint64_t done = 0;
    while (done < info.data_size) {
        const uint64_t n = std::min<uint64_t>(buf.size(), info.data_size - done);
        in.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(n));
        expect(in.gcount() == static_cast<std::streamsize>(n));
        hash = qwen_fnv1a64(luisa::span<const std::byte>{buf.data(), static_cast<size_t>(n)}, hash);
        done += n;
    }
    return hash;
}

uint64_t file_inventory_digest(const luisa::filesystem::path &path,
                               luisa::span<const QwenTensorRef> inventory, uint64_t chunk) {
    uint64_t digest = kFnv1aOffsetBasis;
    for (const auto &ref : inventory) {
        expect(ref.info != nullptr);
        if (ref.info == nullptr) { continue; }
        digest = w4_roll_source_hash(digest, file_payload_hash(path, *ref.info, chunk));
    }
    return digest;
}

bool close_to(double a, double b, double rel = 1e-12) {
    const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) <= rel * scale;
}

void test_w4_source_identity() {
    TempDir tmp;
    const auto file_a = write_tiny_w4_checkpoint(tmp.path / "a", 3u);
    const auto file_b = write_tiny_w4_checkpoint(tmp.path / "b", 9u);
    std::error_code ec;
    // The two checkpoints have the SAME byte size and different weights: the case
    // an identity check that only compares sizes cannot catch.
    expect(luisa::filesystem::file_size(file_a, ec) == luisa::filesystem::file_size(file_b, ec));
    auto ra = GgufReader::open(file_a);
    auto rb = GgufReader::open(file_b);
    expect(ra.ok() && rb.ok());
    if (!ra.ok() || !rb.ok()) { return; }
    auto inv_a = tiny_w4_inventory(**ra.value);
    auto inv_b = tiny_w4_inventory(**rb.value);
    expect(inv_a.size() == 4u);
    expect(inv_b.size() == 4u);
    W4PackOptions options;
    options.clip = W4ClipSearchConfig{128u, 0.85f, 1.0f, 31u};
    const auto pack_dir = tmp.path / "pack_a";
    W4PackReport report;
    expect(pack_w4_from_gguf(**ra.value, tiny_span(inv_a), "tiny_a.gguf", pack_dir, options, &report).ok());
    auto manifest = read_w4_pack_manifest(pack_dir);
    expect(manifest.ok());
    if (!manifest.ok()) { return; }
    const auto &m = *manifest.value;
    expect(m.source_size == (**ra.value).file_size());
    expect(m.source_hash != kFnv1aOffsetBasis);// the digest is really recorded
    expect(m.tensors.size() == 4u);
    // (1) The recorded digest IS the fold of the raw payload hashes: recompute it
    //     here from the FILE with two different chunk sizes.
    for (uint64_t chunk : {64ull * 1024ull, 1ull << 20}) {
        expect(file_inventory_digest(file_a, tiny_span(inv_a), chunk) == m.source_hash);
    }
    // (2) ... and through the public helper the runner uses.
    auto digest_a = w4_source_digest(**ra.value, tiny_span(inv_a), 64ull * 1024ull);
    expect(digest_a.ok());
    if (digest_a.ok()) { expect(*digest_a.value == m.source_hash); }
    auto digest_b = w4_source_digest(**rb.value, tiny_span(inv_b), 64ull * 1024ull);
    expect(digest_b.ok());
    if (digest_b.ok()) { expect(*digest_b.value != m.source_hash); }
    // (3) The verifier accepts the checkpoint the pack came from ...
    expect(verify_w4_pack_source(**ra.value, tiny_span(inv_a), m).ok());
    // (4) ... and REFUSES a same-size checkpoint with different payloads. This is
    //     the regression: the pre-fix code only compared source_size.
    {
        auto st = verify_w4_pack_source(**rb.value, tiny_span(inv_b), m);
        expect(!st.ok());
        expect(st.code == StatusCode::unvalidated);
        expect(st.message.find("SAME file size") != luisa::string::npos);
        expect(st.detail.find("pack=") != luisa::string::npos);
    }
    // (5) Even a ONE-BYTE payload edit in an otherwise identical file is caught.
    {
        auto bytes = [&] {
            std::ifstream in{file_a, std::ios::binary};
            luisa::vector<char> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            return b;
        }();
        const auto *info = (**ra.value).find("blk.0.ffn_gate.weight");
        expect(info != nullptr);
        if (info != nullptr) { bytes[static_cast<size_t>(info->data_offset + 3u)] ^= 0x01; }
        const auto file_c = tmp.path / "flipped.gguf";
        {
            std::ofstream out{file_c, std::ios::binary | std::ios::trunc};
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        expect(luisa::filesystem::file_size(file_c, ec) == luisa::filesystem::file_size(file_a, ec));
        auto rc = GgufReader::open(file_c);
        expect(rc.ok());
        if (rc.ok()) {
            auto inv_c = tiny_w4_inventory(**rc.value);
            auto st = verify_w4_pack_source(**rc.value, tiny_span(inv_c), m);
            expect(!st.ok());
            expect(st.code == StatusCode::unvalidated);
        }
    }
    // (6) A size mismatch and a tampered recorded hash are refused too.
    {
        auto wrong_size = m;
        wrong_size.source_size += 32u;
        auto st = verify_w4_pack_source(**ra.value, tiny_span(inv_a), wrong_size);
        expect(!st.ok() && st.code == StatusCode::unvalidated);
        expect(st.message.find("size mismatch") != luisa::string::npos);
        auto wrong_hash = m;
        wrong_hash.source_hash ^= 1ull;
        auto st2 = verify_w4_pack_source(**ra.value, tiny_span(inv_a), wrong_hash);
        expect(!st2.ok() && st2.code == StatusCode::unvalidated);
    }
    // (7) The rolling rule is the documented one: LE bytes of each per-tensor hash.
    {
        const uint64_t h0 = 0x0102030405060708ull;
        const uint64_t folded = w4_roll_source_hash(kFnv1aOffsetBasis, h0);
        std::byte le[8];
        for (uint32_t b = 0; b < 8u; ++b) { le[b] = static_cast<std::byte>((h0 >> (8u * b)) & 0xFFu); }
        expect(folded == qwen_fnv1a64(luisa::span<const std::byte>{le, 8u}, kFnv1aOffsetBasis));
    }
}

void test_w4_pack_aggregates() {
    TempDir tmp;
    const auto file = write_tiny_w4_checkpoint(tmp.path / "agg", 11u);
    auto reader = GgufReader::open(file);
    expect(reader.ok());
    if (!reader.ok()) { return; }
    auto inventory = tiny_w4_inventory(**reader.value);
    W4PackOptions options;
    options.clip = W4ClipSearchConfig{128u, 0.85f, 1.0f, 31u};
    const auto pack_dir = tmp.path / "pack";
    W4PackReport report;
    expect(pack_w4_from_gguf(**reader.value, tiny_span(inventory), "tiny.gguf", pack_dir, options, &report).ok());
    // Reference aggregation computed from the host packer, per tensor.
    double sse_sum = 0.0, rtn_sum = 0.0, ratio_weighted = 0.0;
    uint64_t elements = 0, groups = 0, clipped = 0;
    luisa::vector<float> ratios;
    for (const char *name : {"blk.0.ffn_gate.weight", "blk.0.ffn_down.weight"}) {
        const auto *info = (**reader.value).find(name);
        expect(info != nullptr);
        if (info == nullptr) { continue; }
        const uint64_t k = info->dims[0];
        const uint64_t n = info->dims[1];
        luisa::vector<float> values(static_cast<size_t>(info->element_count));
        expect((**reader.value).read_tensor_f32(*info, values).ok());
        luisa::vector<uint32_t> words(static_cast<size_t>(info->element_count / 8u), 0u);
        luisa::vector<float> scales(static_cast<size_t>(n * (k / 128u)), 0.0f);
        W4PackStats stats{};
        expect(pack_w4_matrix(values, k, n, options.clip, words, scales, &stats).ok());
        sse_sum += stats.mse * static_cast<double>(k * n);
        rtn_sum += stats.rtn_mse * static_cast<double>(k * n);
        ratio_weighted += static_cast<double>(stats.mean_ratio) * static_cast<double>(stats.groups);
        elements += k * n;
        groups += stats.groups;
        clipped += stats.clipped_groups;
        ratios.emplace_back(stats.mean_ratio);
    }
    expect(elements == 256u * 3u + 128u * 2u);
    expect(groups == 3u * 2u + 2u * 1u);
    expect(report.w4_elements == elements);
    expect(report.groups == groups);
    expect(report.clipped_groups == clipped);
    expect(report.tensors == 4u);
    expect(report.w4_matrices == 2u);
    expect(report.dense_tensors == 2u);
    // Element-weighted MSE, not the raw squared-error sum (the pre-fix bug).
    expect(close_to(report.mse, sse_sum / static_cast<double>(elements), 1e-10));
    expect(close_to(report.rtn_mse, rtn_sum / static_cast<double>(elements), 1e-10));
    expect(std::fabs(report.mse - sse_sum) > std::fabs(report.mse));
    // The unweighted arm reports the same objective in both fields.
    expect(close_to(report.weighted_mse, report.mse, 1e-10));
    expect(close_to(report.rtn_weighted_mse, report.rtn_mse, 1e-10));
    // Group-weighted mean ratio over BOTH matrices, not the last tensor's ratio.
    const auto expected_ratio = static_cast<float>(ratio_weighted / static_cast<double>(groups));
    expect(close_to(report.mean_ratio, expected_ratio, 1e-9));
    expect(ratios.size() == 2u);
    if (ratios.size() == 2u) {
        expect(ratios[0] != ratios[1]);// the fixture must expose the difference
        expect(report.mean_ratio != ratios[1]);
    }
    // The improvement ratio stays valid and is derived from the aggregates.
    expect(report.rtn_mse > 0.0);
    expect(close_to(report.mse_improvement(), 100.0 * (1.0 - report.mse / report.rtn_mse), 1e-12));
    // Deterministic reset: a caller-supplied report never inherits stale numbers.
    {
        W4PackReport dirty;
        dirty.tensors = 999u;
        dirty.w4_elements = 12345u;
        dirty.groups = 77u;
        dirty.clipped_groups = 5u;
        dirty.mse = 1.0;
        dirty.rtn_mse = 2.0;
        dirty.mean_ratio = 0.5f;
        const auto pack_dir2 = tmp.path / "pack2";
        expect(pack_w4_from_gguf(**reader.value, tiny_span(inventory), "tiny.gguf", pack_dir2, options, &dirty).ok());
        expect(dirty.tensors == report.tensors);
        expect(dirty.w4_elements == report.w4_elements);
        expect(dirty.groups == report.groups);
        expect(dirty.clipped_groups == report.clipped_groups);
        expect(dirty.mse == report.mse);
        expect(dirty.rtn_mse == report.rtn_mse);
        expect(dirty.mean_ratio == report.mean_ratio);
    }
}

// The weighted clipping objective (opt_plan/05 section 3 item 2):
//   E_g(s) = sum_{j in g} m[j] (v[j] - q_j(s) s)^2 / sum_{j in g} m[j]
void test_w4_importance_weighting() {
    constexpr uint64_t K = 256u;
    constexpr uint64_t N = 2u;
    W4ClipSearchConfig cfg{128u, 0.85f, 1.0f, 31u};
    // One strong outlier per group: clipping trades the bulk against the outlier,
    // so shifting the importance between them must move the chosen ratio.
    luisa::vector<float> w(K * N, 0.0f);
    for (uint64_t n = 0; n < N; ++n) {
        for (uint64_t k = 0; k < K; ++k) {
            const uint64_t j = k % 128u;
            const float tiny = std::sin(static_cast<float>(k) * 0.37f + static_cast<float>(n)) * 0.02f;
            // 1 outlier at 4.0, 8 channels at +-1.0, everything else tiny. With this
            // balance the CLIPPING-ONLY optimum is an interior grid ratio (~0.95),
            // so an importance shift in either direction must move it: protecting
            // the outlier raises it towards 1.0, protecting the +-1.0 bulk lowers it.
            float value = tiny;
            if (j == 7u) {
                value = 4.0f;
            } else if (j == 20u || j == 40u || j == 60u || j == 80u || j == 100u || j == 110u ||
                       j == 120u || j == 127u) {
                value = ((j / 20u) % 2u == 0u) ? 1.0f : -1.0f;
            }
            w[static_cast<size_t>(n * K + k)] = value;
        }
    }
    const auto run = [&](luisa::span<const float> importance, luisa::vector<uint32_t> &words,
                         luisa::vector<float> &scales, W4PackStats *stats) {
        words.assign(static_cast<size_t>(K * N / 8u), 0u);
        scales.assign(static_cast<size_t>(N * (K / 128u)), 0.0f);
        return pack_w4_matrix_weighted(w, importance, K, N, cfg, words, scales, stats);
    };
    // (1) Uniform weights reproduce the clipping-only search BYTE FOR BYTE.
    {
        luisa::vector<float> uniform(static_cast<size_t>(K), 1.0f);
        luisa::vector<uint32_t> wi, wu;
        luisa::vector<float> si, su;
        W4PackStats st_i{}, st_u{};
        expect(run(uniform, wi, si, &st_i).ok());
        // The clipping-only search is the weighted search with an EMPTY importance
        // span: it must be byte-identical (a constant weight vector cancels).
        expect(run(luisa::span<const float>{}, wu, su, &st_u).ok());
        expect(wi.size() == wu.size() && si.size() == su.size());
        expect(std::memcmp(wi.data(), wu.data(), wi.size() * 4u) == 0);
        expect(std::memcmp(si.data(), su.data(), si.size() * 4u) == 0);
        expect(st_i.mse == st_u.mse && st_i.rtn_mse == st_u.rtn_mse);
        expect(st_i.clipped_groups == st_u.clipped_groups);
    }
    // (2) The chosen scale moves in the expected direction: protecting the outlier
    //     keeps the ratio high (clip less), protecting the bulk lowers it (clip
    //     more).
    luisa::vector<float> outlier_heavy(static_cast<size_t>(K), 1e-3f);
    luisa::vector<float> bulk_heavy(static_cast<size_t>(K), 1.0f);
    for (uint64_t k = 0; k < K; ++k) {
        const bool outlier = (k % 128u == 7u);
        outlier_heavy[static_cast<size_t>(k)] = outlier ? 1.0f : 1e-3f;
        bulk_heavy[static_cast<size_t>(k)] = outlier ? 1e-3f : 1.0f;
    }
    luisa::vector<float> uniform(static_cast<size_t>(K), 1.0f);
    luisa::vector<uint32_t> w_u, w_o, w_b;
    luisa::vector<float> s_u, s_o, s_b;
    W4PackStats st_u{}, st_o{}, st_b{};
    expect(run(uniform, w_u, s_u, &st_u).ok());
    expect(run(outlier_heavy, w_o, s_o, &st_o).ok());
    expect(run(bulk_heavy, w_b, s_b, &st_b).ok());
    expect(st_b.mean_ratio < st_u.mean_ratio);
    expect(st_u.mean_ratio < st_o.mean_ratio);
    expect(st_b.mean_ratio < st_o.mean_ratio);
    // The codes/scales really change in BOTH directions (a weighting that changed
    // nothing would be a placeholder).
    expect(std::memcmp(s_u.data(), s_o.data(), s_u.size() * 4u) != 0);
    expect(std::memcmp(s_u.data(), s_b.data(), s_u.size() * 4u) != 0);
    // The DECODED weights therefore differ too. (The 4-bit CODES of this fixture
    // happen to survive the scale shift -- only three magnitudes exist -- which is
    // exactly why the search result must be judged on the scales/objective, not on
    // "the words changed".)
    luisa::vector<float> dq_u(K * N), dq_o(K * N);
    expect(dequantize_w4_matrix(w_u, s_u, K, N, 128u, dq_u).ok());
    expect(dequantize_w4_matrix(w_o, s_o, K, N, 128u, dq_o).ok());
    expect(std::memcmp(dq_u.data(), dq_o.data(), dq_u.size() * 4u) != 0);
    // (3) The plain per-element MSE stays a plain MSE, and the weighted objective
    //     is what the search minimized (never larger than the RTN weighted value).
    expect(st_o.mse > 0.0 && st_o.rtn_mse > 0.0);
    // The search minimizes the error of the FLOAT-rounded stored scale while the RTN
    // reference uses the exact double scale, so allow a relative epsilon at a tie.
    expect(st_o.weighted_mse <=
           st_o.rtn_weighted_mse + 1e-6 * st_o.rtn_mse + 1e-30);
    // (4) Reproducibility: the same inputs give byte-identical words and scales.
    {
        luisa::vector<uint32_t> w2;
        luisa::vector<float> s2;
        expect(run(outlier_heavy, w2, s2, nullptr).ok());
        expect(std::memcmp(w_o.data(), w2.data(), w_o.size() * 4u) == 0);
        expect(std::memcmp(s_o.data(), s2.data(), s_o.size() * 4u) == 0);
    }
    // (5) Fail closed on unusable statistics: wrong length, NaN, negative, a group
    //     with zero total weight.
    {
        luisa::vector<uint32_t> words;
        luisa::vector<float> scales;
        luisa::vector<float> short_vec(static_cast<size_t>(K - 8u), 1.0f);
        expect(!run(short_vec, words, scales, nullptr).ok());
        luisa::vector<float> nan_vec(static_cast<size_t>(K), 1.0f);
        nan_vec[3] = std::nanf("");
        expect(!run(nan_vec, words, scales, nullptr).ok());
        luisa::vector<float> neg_vec(static_cast<size_t>(K), 1.0f);
        neg_vec[5] = -1.0f;
        expect(!run(neg_vec, words, scales, nullptr).ok());
        luisa::vector<float> zero_group(static_cast<size_t>(K), 1.0f);
        for (uint64_t k = 0; k < 128u; ++k) { zero_group[static_cast<size_t>(k)] = 0.0f; }
        expect(!run(zero_group, words, scales, nullptr).ok());
    }
}

QwenCalibrationReport make_calibration_report(luisa::span<const QwenTensorRef> inventory,
                                              uint64_t tokens, float outlier_boost) {
    QwenCalibrationReport report;
    report.tokens = tokens;
    report.projections = 0;
    for (const auto &ref : inventory) {
        if (ref.info == nullptr) { continue; }
        // Only the tensors this pack would QUANTIZE get statistics (a dense F32
        // tensor has no 4-bit group objective to weight).
        auto storage = classify_w4_pack_storage(ref.info->name, ref.info->dims, 128u);
        if (!storage.ok() || *storage.value != WeightStorage::w4_packed) { continue; }
        QwenCalibrationTensor t;
        t.name = ref.info->name;
        t.k = ref.info->dims[0];
        t.tokens = tokens;
        t.second_moment.resize(static_cast<size_t>(t.k), 0.0f);
        for (uint64_t j = 0; j < t.k; ++j) {
            // Strongly skewed, deterministic "measured" second moments.
            const bool hot = (j % 17u) == 0u;
            t.second_moment[static_cast<size_t>(j)] = hot ? outlier_boost : 1.0f;
        }
        report.observed.push_back(std::move(t));
        report.projections += 2u;
    }
    report.tensors = report.observed.size();
    return report;
}

void test_w4_importance_file() {
    TempDir tmp;
    const auto file = write_tiny_w4_checkpoint(tmp.path / "imp", 5u);
    auto reader = GgufReader::open(file);
    expect(reader.ok());
    if (!reader.ok()) { return; }
    auto inventory = tiny_w4_inventory(**reader.value);
    auto digest = w4_source_digest(**reader.value, tiny_span(inventory), 64ull * 1024ull);
    expect(digest.ok());
    if (!digest.ok()) { return; }
    W4ImportanceSource source;
    source.name = "tiny_5.gguf";
    source.size = (**reader.value).file_size();
    source.hash = *digest.value;
    source.prompts = 2u;
    source.prompt_len = 8u;
    source.decode_tokens = 4u;
    source.tokens = 24u;
    const auto calib = make_calibration_report(tiny_span(inventory), 24u, 200.0f);
    expect(calib.observed.size() == 2u);
    const auto dir = tmp.path / "importance";
    // A calibration that never exercised a tensor is refused (no invented data).
    {
        auto broken = calib;
        broken.observed[0].tokens = 0u;
        expect(!write_w4_importance(dir, source, broken).ok());
    }
    expect(write_w4_importance(dir, source, calib).ok());
    auto stats = read_w4_importance(dir);
    expect(stats.ok());
    if (!stats.ok()) { return; }
    const auto &s = *stats.value;
    expect(s.version == kW4ImportanceVersion);
    expect(s.source.hash == source.hash);
    expect(s.source.size == source.size);
    expect(s.source.prompts == 2u && s.source.prompt_len == 8u && s.source.decode_tokens == 4u);
    expect(s.source.tokens == 24u);
    expect(s.tensors.size() == 2u);
    expect(s.values.size() == 256u + 128u);
    // Lookup is bound by name AND K.
    expect(s.channels("blk.0.ffn_gate.weight", 256u).size() == 256u);
    expect(s.channels("blk.0.ffn_gate.weight", 128u).empty());
    expect(s.channels("blk.0.ffn_down.weight", 128u).size() == 128u);
    expect(s.channels("blk.0.missing.weight", 128u).empty());
    expect(s.status_for("blk.0.missing.weight", 128u).code == StatusCode::not_found);
    expect(s.status_for("blk.0.ffn_gate.weight", 128u).code == StatusCode::unvalidated);
    expect(s.status_for("blk.0.ffn_gate.weight", 256u).ok());
    // The recorded summary describes the payload, and the values are the ones the
    // calibration reported.
    const auto channels = s.channels("blk.0.ffn_gate.weight", 256u);
    expect(channels[0] == 200.0f);
    expect(channels[1] == 1.0f);
    expect(s.tensors[0].mean > 1.0f && s.tensors[0].max == 200.0f);
    // A tampered payload byte is refused (stale/corrupt calibration).
    {
        const auto payload = dir / kW4ImportancePayloadName;
        std::error_code ec;
        const auto bytes = luisa::filesystem::file_size(payload, ec);
        expect(bytes == (256u + 128u) * 4u);
        {
            std::fstream f{payload, std::ios::binary | std::ios::in | std::ios::out};
            expect(f.is_open());
            char byte = 0;
            f.seekg(0);
            f.read(&byte, 1);
            byte = static_cast<char>(byte ^ 0x02);
            f.seekp(0);
            f.write(&byte, 1);
        }
        auto bad = read_w4_importance(dir);
        expect(!bad.ok());
        expect(bad.status.code == StatusCode::unvalidated);
        // restore by rewriting the whole file
        expect(write_w4_importance(dir, source, calib).ok());
        expect(read_w4_importance(dir).ok());
    }
    // A stats file for a different (same-size) checkpoint is detected by the
    // digest, not by the size.
    {
        auto other = source;
        other.hash ^= 1ull;
        const auto other_dir = tmp.path / "importance_stale";
        expect(write_w4_importance(other_dir, other, calib).ok());
        auto other_stats = read_w4_importance(other_dir);
        expect(other_stats.ok());
        if (other_stats.ok()) {
            expect(other_stats.value->source.size == source.size);
            expect(other_stats.value->source.hash != source.hash);
        }
    }
    // Non-finite / all-zero statistics are refused at write time.
    {
        auto bad = calib;
        bad.observed[1].second_moment[4] = std::nanf("");
        expect(!write_w4_importance(tmp.path / "importance_nan", source, bad).ok());
        auto zero = calib;
        for (auto &v : zero.observed[0].second_moment) { v = 0.0f; }
        expect(!write_w4_importance(tmp.path / "importance_zero", source, zero).ok());
        auto negative = calib;
        negative.observed[0].second_moment[9] = -3.0f;
        expect(!write_w4_importance(tmp.path / "importance_neg", source, negative).ok());
    }
    // A missing payload is a hard failure, and so is an empty directory.
    {
        const auto broken_dir = tmp.path / "importance_broken";
        std::error_code ec;
        luisa::filesystem::create_directories(broken_dir, ec);
        expect(!read_w4_importance(broken_dir).ok());
        expect(write_w4_importance(broken_dir, source, calib).ok());
        luisa::filesystem::remove(broken_dir / kW4ImportancePayloadName, ec);
        auto missing = read_w4_importance(broken_dir);
        expect(!missing.ok());
        expect(missing.status.code == StatusCode::not_found);
    }
}

void test_w4_importance_aware_pack() {
    TempDir tmp;
    const auto file = write_tiny_w4_checkpoint(tmp.path / "packckpt", 7u);
    auto reader = GgufReader::open(file);
    expect(reader.ok());
    if (!reader.ok()) { return; }
    auto inventory = tiny_w4_inventory(**reader.value);
    auto digest = w4_source_digest(**reader.value, tiny_span(inventory), 64ull * 1024ull);
    expect(digest.ok());
    if (!digest.ok()) { return; }
    W4ImportanceSource source;
    source.name = "tiny_7.gguf";
    source.size = (**reader.value).file_size();
    source.hash = *digest.value;
    source.prompts = 2u;
    source.prompt_len = 8u;
    source.decode_tokens = 4u;
    source.tokens = 24u;
    const auto calib = make_calibration_report(tiny_span(inventory), 24u, 200.0f);
    const auto importance_dir = tmp.path / "importance";
    expect(write_w4_importance(importance_dir, source, calib).ok());
    auto stats = read_w4_importance(importance_dir);
    expect(stats.ok());
    if (!stats.ok()) { return; }
    W4PackOptions plain;
    plain.clip = W4ClipSearchConfig{128u, 0.85f, 1.0f, 31u};
    W4PackOptions weighted = plain;
    weighted.importance = &*stats.value;
    weighted.full_inventory = true;// this inventory IS the checkpoint's inventory
    const auto plain_dir = tmp.path / "pack_plain";
    const auto weighted_dir = tmp.path / "pack_weighted";
    W4PackReport plain_report, weighted_report;
    expect(pack_w4_from_gguf(**reader.value, tiny_span(inventory), "tiny_7.gguf", plain_dir, plain,
                             &plain_report).ok());
    expect(pack_w4_from_gguf(**reader.value, tiny_span(inventory), "tiny_7.gguf", weighted_dir,
                             weighted, &weighted_report).ok());
    auto weighted_manifest = read_w4_pack_manifest(weighted_dir);
    expect(weighted_manifest.ok());
    if (!weighted_manifest.ok()) { return; }
    // The manifest records the calibration (provenance + token count + digest).
    expect(weighted_manifest.value->has_importance);
    expect(weighted_manifest.value->importance_tokens == 24u);
    expect(weighted_manifest.value->importance_source_hash == source.hash);
    expect(weighted_manifest.value->importance_name == source.name);
    expect(weighted_manifest.value->importance_hash != 0u);
    auto plain_manifest = read_w4_pack_manifest(plain_dir);
    expect(plain_manifest.ok() && !plain_manifest.value->has_importance);
    // Coverage: every W4 matrix was weighted, none fell back silently.
    expect(weighted_report.importance_matrices == 2u);
    expect(weighted_report.unweighted_matrices == 0u);
    expect(weighted_report.w4_matrices == 2u);
    // The weighted objective is reported for the weighted arm only.
    expect(weighted_report.weighted_mse > 0.0);
    expect(weighted_report.rtn_weighted_mse > 0.0);
    // ... and the quantization actually CHANGED (not a placeholder).
    const auto read_payload = [](const luisa::filesystem::path &dir, uint64_t index,
                                 const char *suffix) {
        const auto path = dir / "tensors" /
                          luisa::filesystem::path{std::to_string(index) + suffix};
        std::ifstream in{path, std::ios::binary};
        return luisa::vector<char>((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    };
    for (const char *suffix : {".w4.bin", ".w4s.bin"}) {
        const auto a = read_payload(plain_dir, 0u, suffix);
        const auto b = read_payload(weighted_dir, 0u, suffix);
        expect(!a.empty() && a.size() == b.size());
        expect(std::memcmp(a.data(), b.data(), a.size()) != 0);
    }
    // Fail closed: statistics from a same-size/different-payload checkpoint.
    {
        auto stale = *stats.value;
        stale.source.hash ^= 1ull;
        auto options = plain;
        options.importance = &stale;
        options.full_inventory = true;
        auto st = pack_w4_from_gguf(**reader.value, tiny_span(inventory), "tiny_7.gguf",
                                    tmp.path / "pack_stale", options, nullptr);
        expect(!st.ok());
        expect(st.code == StatusCode::unvalidated);
        expect(st.message.find("SAME file size") != luisa::string::npos);
    }
    // Fail closed: a size mismatch is caught even before the digest.
    {
        auto stale = *stats.value;
        stale.source.size += 8u;
        auto options = plain;
        options.importance = &stale;
        auto st = pack_w4_from_gguf(**reader.value, tiny_span(inventory), "tiny_7.gguf",
                                    tmp.path / "pack_stale2", options, nullptr);
        expect(!st.ok() && st.code == StatusCode::unvalidated);
        expect(st.message.find("source size mismatch") != luisa::string::npos);
    }
    // Fail closed: statistics that do not cover every matrix (named tensor).
    {
        auto partial = *stats.value;
        partial.tensors.pop_back();
        auto options = plain;
        options.importance = &partial;
        auto st = pack_w4_from_gguf(**reader.value, tiny_span(inventory), "tiny_7.gguf",
                                    tmp.path / "pack_missing", options, nullptr);
        expect(!st.ok());
        expect(st.code == StatusCode::not_found);
        expect(st.detail.find("blk.0.") != luisa::string::npos);
    }
    // Fail closed: statistics collected for a different K.
    {
        auto wrong_k = *stats.value;
        wrong_k.tensors[0].k = 128u;
        auto options = plain;
        options.importance = &wrong_k;
        auto st = pack_w4_from_gguf(**reader.value, tiny_span(inventory), "tiny_7.gguf",
                                    tmp.path / "pack_k", options, nullptr);
        expect(!st.ok());
        expect(st.code == StatusCode::unvalidated);
        expect(st.detail.find("stats_k=128") != luisa::string::npos);
    }
}

}// namespace

// ---------------------------------------------------------------------------
// OFFLINE merged projection pack -- HOST contract (opt_plan/03 section 3 + D5).
//
// No device and no checkpoint: a hand-built two-matrix inventory plus a manifest
// exercise the writer/reader, the STRUCTURAL gate (kind/segment roles/rows/order/
// K/N/format/group), the payload byte-size + hash gate and the fail-closed
// mutations a corrupt or foreign pack would present. The end-to-end packer,
// source-identity and upload-replacement checks live in the device arm below,
// because they need a validated model inventory.
// ---------------------------------------------------------------------------
void test_projection_pack_manifest() {
    // One GDN-style and one attention-style matrix, deliberately tiny but exact.
    using R = QwenTensorRole;
    luisa::vector<QwenMergedMatrixSpec> inventory(2u);
    {
        auto &m = inventory[0];
        m.kind = QwenMergeKind::gdn_qkv_z_ab;
        m.layer = 0u;
        m.k = 8u;
        const uint64_t rows[] = {8u, 4u, 1u, 1u};
        const R roles[] = {R::gdn_qkv, R::gdn_gate, R::gdn_alpha, R::gdn_beta};
        uint64_t off = 0u;
        for (uint32_t i = 0; i < 4u; i++) {
            QwenTensorSegment seg;
            seg.role = roles[i];
            seg.name = qwen_tensor_name(roles[i], m.layer);
            seg.rows = rows[i];
            seg.row_offset = off;
            off += rows[i];
            m.segments.emplace_back(std::move(seg));
        }
        m.n = off;// 14
    }
    {
        auto &m = inventory[1];
        m.kind = QwenMergeKind::attn_qkv;
        m.layer = 3u;
        m.k = 8u;
        const uint64_t rows[] = {8u, 1u, 1u};
        const R roles[] = {R::attn_q, R::attn_k, R::attn_v};
        uint64_t off = 0u;
        for (uint32_t i = 0; i < 3u; i++) {
            QwenTensorSegment seg;
            seg.role = roles[i];
            seg.name = qwen_tensor_name(roles[i], m.layer);
            seg.rows = rows[i];
            seg.row_offset = off;
            off += rows[i];
            m.segments.emplace_back(std::move(seg));
        }
        m.n = off;// 10
    }
    // Manifest that matches the inventory, plus two payload files whose hashes are
    // computed from the bytes actually written.
    TempDir tmp;
    std::error_code ec;
    luisa::filesystem::create_directories(tmp.path / "matrices", ec);
    ProjectionPackManifest manifest;
    manifest.version = kProjectionPackVersion;
    manifest.source_name = "synthetic-checkpoint.gguf";
    manifest.source_size = 4096u;
    manifest.source_hash = 0x0123456789abcdefull;
    manifest.storage = WeightStorage::bf16_packed;
    manifest.group = 0u;
    manifest.elements_per_word = 2u;
    manifest.scales_per_row = 0u;
    manifest.format_order = "bf16";
    luisa::vector<luisa::vector<std::byte>> payloads;
    for (size_t i = 0; i < inventory.size(); ++i) {
        const auto &spec = inventory[i];
        ProjectionPackMatrixRecord rec;
        rec.name = projection_pack_matrix_name(spec.kind, spec.layer);
        rec.kind = spec.kind;
        rec.layer = spec.layer;
        rec.k = spec.k;
        rec.n = spec.n;
        rec.storage = manifest.storage;
        rec.group = 0u;
        rec.scales_per_row = 0u;
        for (const auto &seg : spec.segments) {
            auto with_source = seg;
            with_source.source_bytes = seg.rows * spec.k * 2u;
            with_source.source_hash = 0x1000u + seg.row_offset;
            rec.segments.emplace_back(with_source);
        }
        const uint64_t bytes = spec.n * spec.k / 2u * 4u;
        luisa::vector<std::byte> payload(static_cast<size_t>(bytes));
        for (size_t b = 0; b < payload.size(); ++b) {
            payload[b] = static_cast<std::byte>((b * 37u + i * 11u) & 0xFFu);
        }
        rec.word_bytes = bytes;
        rec.word_hash = qwen_fnv1a64(payload);
        rec.scale_bytes = 0u;
        rec.scale_hash = 0u;
        payloads.emplace_back(std::move(payload));
        manifest.matrices.emplace_back(std::move(rec));
    }
    // Geometry helper contracts.
    expect(projection_pack_elements_per_word(WeightStorage::bf16_packed).ok() &&
           *projection_pack_elements_per_word(WeightStorage::bf16_packed).value == 2u);
    expect(projection_pack_elements_per_word(WeightStorage::fp8_packed).ok() &&
           *projection_pack_elements_per_word(WeightStorage::fp8_packed).value == 4u);
    expect(!projection_pack_elements_per_word(WeightStorage::w4_packed).ok());// W4 is not mergeable
    expect(projection_pack_scales_per_row(WeightStorage::bf16_packed, 0u, 2560u).ok() &&
           *projection_pack_scales_per_row(WeightStorage::bf16_packed, 0u, 2560u).value == 0u);
    expect(projection_pack_scales_per_row(WeightStorage::fp8_packed, 64u, 2560u).ok() &&
           *projection_pack_scales_per_row(WeightStorage::fp8_packed, 64u, 2560u).value == 40u);
    expect(!projection_pack_scales_per_row(WeightStorage::fp8_packed, 32u, 2560u).ok());
    expect(!projection_pack_scales_per_row(WeightStorage::fp8_packed, 64u, 2558u).ok());
    expect(projection_pack_scales_per_row(WeightStorage::q8_0_packed, 32u, 2560u).ok() &&
           *projection_pack_scales_per_row(WeightStorage::q8_0_packed, 32u, 2560u).value == 80u);
    expect(!projection_pack_scales_per_row(WeightStorage::q8_0_packed, 32u, 2559u).ok());

    // The structural gate accepts the consistent pair.
    expect(validate_projection_pack_manifest(manifest, inventory).ok());

    // Write the payload files, then the manifest LAST (the pack order).
    for (size_t i = 0; i < payloads.size(); ++i) {
        luisa::filesystem::path words_path, scales_path;
        expect(projection_pack_payload_paths(tmp.path, i, words_path, scales_path).ok());
        write_file(words_path, payloads[i]);
    }
    expect(write_projection_pack_manifest(tmp.path, manifest).ok());
    // Payloads verify (size + FNV-1a 64) ...
    expect(verify_projection_pack_payloads(tmp.path, manifest).ok());
    // ... and a single flipped payload byte is refused.
    {
        luisa::filesystem::path words_path, scales_path;
        expect(projection_pack_payload_paths(tmp.path, 1u, words_path, scales_path).ok());
        auto copy = payloads[1];
        copy[3] = static_cast<std::byte>(static_cast<uint8_t>(copy[3]) ^ 0x5Au);
        write_file(words_path, copy);
        expect(!verify_projection_pack_payloads(tmp.path, manifest).ok());
        write_file(words_path, payloads[1]);
        expect(verify_projection_pack_payloads(tmp.path, manifest).ok());
    }
    // A truncated payload is refused too (size gate).
    {
        luisa::filesystem::path words_path, scales_path;
        expect(projection_pack_payload_paths(tmp.path, 0u, words_path, scales_path).ok());
        write_file(words_path, luisa::span<const std::byte>{payloads[0].data(),
                                                            payloads[0].size() - 4u});
        expect(!verify_projection_pack_payloads(tmp.path, manifest).ok());
        write_file(words_path, payloads[0]);
        expect(verify_projection_pack_payloads(tmp.path, manifest).ok());
    }

    // Manifest round trip: every recorded field survives the text form.
    {
        auto back = read_projection_pack_manifest(tmp.path);
        expect(back.ok());
        if (back.ok()) {
            const auto &b = *back.value;
            expect(b.version == manifest.version);
            expect(b.source_name == manifest.source_name);
            expect(b.source_size == manifest.source_size);
            expect(b.source_hash == manifest.source_hash);
            expect(b.storage == manifest.storage);
            expect(b.elements_per_word == manifest.elements_per_word);
            expect(b.scales_per_row == manifest.scales_per_row);
            expect(b.matrices.size() == manifest.matrices.size());
            for (size_t i = 0; i < manifest.matrices.size(); ++i) {
                const auto &x = manifest.matrices[i];
                const auto &y = b.matrices[i];
                expect(y.name == x.name);
                expect(y.kind == x.kind);
                expect(y.layer == x.layer);
                expect(y.k == x.k);
                expect(y.n == x.n);
                expect(y.word_bytes == x.word_bytes);
                expect(y.word_hash == x.word_hash);
                expect(y.segments.size() == x.segments.size());
                for (size_t s = 0; s < x.segments.size(); ++s) {
                    expect(y.segments[s].role == x.segments[s].role);
                    expect(y.segments[s].name == x.segments[s].name);
                    expect(y.segments[s].rows == x.segments[s].rows);
                    expect(y.segments[s].row_offset == x.segments[s].row_offset);
                    expect(y.segments[s].source_hash == x.segments[s].source_hash);
                }
            }
            if (auto st = validate_projection_pack_manifest(b, inventory); !st.ok()) {
              std::printf("round-trip manifest rejected: %s (%s)\n",
                          st.message.c_str(), st.detail.c_str());
          }
          expect(validate_projection_pack_manifest(b, inventory).ok());
        }
    }

    // Fail-closed mutations of a valid manifest (each must be refused).
    const auto refuses = [&](ProjectionPackManifest bad, const char *what) {
      const bool refused = !validate_projection_pack_manifest(bad, inventory).ok();
      if (!refused) { std::printf("mutation NOT refused: %s\n", what); }
      expect(refused);
    };
    {
        auto bad = manifest;
        bad.matrices[0].n += 1u;// geometry no longer covers N
        refuses(bad, "merged pack geometry refuses a wrong N");
    }
    {
        auto bad = manifest;
        std::swap(bad.matrices[0].segments[0].rows, bad.matrices[0].segments[1].rows);
        refuses(bad, "merged pack geometry refuses a re-segmented matrix");
    }
    {
        auto bad = manifest;
        bad.matrices[0].segments[2].role = R::attn_k;// wrong role keeps the right rows
        refuses(bad, "merged pack geometry refuses a segment role mismatch");
    }
    {
        auto bad = manifest;
        bad.matrices[1].kind = QwenMergeKind::gdn_qkv_z_ab;
        refuses(bad, "merged pack geometry refuses a wrong kind");
    }
    {
        auto bad = manifest;
        bad.matrices[0].k = 16u;
        refuses(bad, "merged pack geometry refuses a wrong K");
    }
    {
        auto bad = manifest;
        bad.storage = WeightStorage::fp8_packed;// pack format says fp8, records say bf16
        refuses(bad, "merged pack geometry refuses a format mismatch");
    }
    {
        auto bad = manifest;
        bad.matrices[0].segments[1].source_hash = 0u;// identity missing
        refuses(bad, "merged pack geometry refuses a segment without source identity");
    }
    {
        auto bad = manifest;
        bad.matrices[0].word_bytes -= 4u;// payload size inconsistent with geometry
        refuses(bad, "merged pack geometry refuses an inconsistent payload size");
    }
    {
        auto bad = manifest;
        bad.matrices.pop_back();// missing matrix
        refuses(bad, "merged pack geometry refuses an incomplete inventory");
    }
    // A manifest that is not a projection pack at all is refused by the reader.
    {
        TempDir other;
        write_file(other.path / "manifest.txt",
                   luisa::span<const std::byte>{});
        auto r = read_projection_pack_manifest(other.path);
        expect(!r.ok());
    }
}

static auto test_qwen_weights_registration = [] {
    "qwen_bf16_codec"_test = [] { test_bf16_codec(); };
    "qwen_fp8_codec"_test = [] { test_fp8_codec(); };
    "qwen_packing"_test = [] { test_packing(); };
    "qwen_fp8_quantize"_test = [] { test_fp8_quantize(); };
    "qwen_classification"_test = [] { test_classification(); };
    "qwen_projection_pack_manifest"_test = [] { test_projection_pack_manifest(); };
    "qwen_w4_geometry"_test = [] { test_w4_geometry(); };
    "qwen_w4_pack_roundtrip"_test = [] { test_w4_pack_roundtrip(); };
    "qwen_w4_dsl_parity"_test = [] { test_w4_dsl_parity(); };
    "qwen_w4_pack_manifest"_test = [] { test_w4_pack_manifest(); };
    "qwen_w4_pack_upload"_test = [] { test_w4_pack_upload(); };
    "qwen_w4_source_identity"_test = [] { test_w4_source_identity(); };
    "qwen_w4_pack_aggregates"_test = [] { test_w4_pack_aggregates(); };
    "qwen_w4_importance_weighting"_test = [] { test_w4_importance_weighting(); };
    "qwen_w4_importance_file"_test = [] { test_w4_importance_file(); };
    "qwen_w4_importance_pack"_test = [] { test_w4_importance_aware_pack(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    g_argc = argc;
    g_argv = const_cast<const char **>(argv);
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    // Register the device test here (not at static init) so it sees argc/argv.
    "qwen_device_upload"_test = [] { test_device_upload(); };
    "qwen_device_upload_q8_real"_test = [] { test_device_upload_q8_real(); };
}
