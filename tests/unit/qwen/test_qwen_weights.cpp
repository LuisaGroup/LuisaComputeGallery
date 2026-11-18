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
#include "qwen/gguf.h"
#include "qwen/model.h"
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
    // Q8_0 explicitly rejected
    auto q = classify_weight_storage("blk.0.ffn_gate.weight", 8u, d2, true);
    expect(!q.ok());
    expect(q.status.code == StatusCode::unsupported);
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

// Deterministic synthetic value for tensor `salt`, element i. Small magnitudes
// keep BF16 rounding exact-ish and FP8 quantization well-conditioned; the
// per-tensor salt makes cross-tensor mix-ups detectable.
float synth_value(uint32_t salt, uint64_t i) {
    return std::sin(static_cast<float>(i % 4096u) * 0.017f + static_cast<float>(salt) * 0.31f) * 0.25f;
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
}

}// namespace

static auto test_qwen_weights_registration = [] {
    "qwen_bf16_codec"_test = [] { test_bf16_codec(); };
    "qwen_fp8_codec"_test = [] { test_fp8_codec(); };
    "qwen_packing"_test = [] { test_packing(); };
    "qwen_fp8_quantize"_test = [] { test_fp8_quantize(); };
    "qwen_classification"_test = [] { test_classification(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    g_argc = argc;
    g_argv = const_cast<const char **>(argv);
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    // Register the device test here (not at static init) so it sees argc/argv.
    "qwen_device_upload"_test = [] { test_device_upload(); };
}
