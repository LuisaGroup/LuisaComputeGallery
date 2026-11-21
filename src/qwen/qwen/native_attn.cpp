// Qwen3.5-4B causal GQA prefill attention through the CUDA native-shader route
// (opt_plan/09 N1b) -- implementation. See native_attn.h for the contract and
// native_attn_kernel.h for the device code.
//
// There is no DSL, no AST and no backend plugin here: the device code is a CUDA
// C++ translation unit that NVRTC compiles at run time (through the CUDA backend's
// NativeShaderExtension) and the launcher binds the five buffers plus five scalars
// by reflection index. That is also why this file only ever reports failures:
// every step (extension lookup, NVRTC, reflection, module load) is a hard gate that
// leaves the object invalid rather than producing a half-working accelerator.
#include "qwen/native_attn.h"

#include <algorithm>
#include <cmath>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/command.h>

#include "qwen/native_attn_kernel.h"

namespace luisa::qwen::dsl {

namespace {

using compute::NativeShader;
using compute::NativeShaderCompileInfo;
using compute::NativeShaderExt;
using compute::NativeShaderLanguage;
using compute::NativeShaderSourceType;
using compute::Usage;

// Entry point of the generated `__global__` (see native_attn_kernel.h).
constexpr auto qwen_native_attn_entry = luisa::string_view{"qwen_native_attn"};

// Parameters of the device kernel, in declaration order: five buffer bindings
// followed by five scalar kernel parameters (tokens, scale, pos0, kv_len, capacity).
constexpr uint32_t qwen_native_attn_bindings = 5u;
constexpr uint32_t qwen_native_attn_scalar_bytes = 4u * sizeof(uint32_t) + sizeof(float);

// sm_80+ per-block STATIC shared-memory ceiling. The native launch path requests no
// dynamic shared memory (cuda_shader_native.cpp passes 0), so the kernel tile must
// fit this; the gate below is what keeps a future re-tiling honest.
constexpr uint32_t qwen_native_attn_static_smem_limit = 48u * 1024u;

[[nodiscard]] Status err(StatusCode code, luisa::string msg, luisa::string detail = {}) {
    return Status::error(code, std::move(msg), std::move(detail));
}

void substitute(luisa::string &text, luisa::string_view token, uint32_t value) noexcept {
    auto replacement = luisa::format("{}", value);
    for (auto pos = text.find(token); pos != luisa::string::npos;
         pos = text.find(token, pos + replacement.size())) {
        text.replace(pos, token.size(), replacement);
    }
}

}// namespace

uint32_t qwen_native_attn_shared_bytes(const QwenNativeAttnConfig &cfg) noexcept {
    if (cfg.head_dim == 0u || cfg.kv_tile == 0u || cfg.q_tile == 0u) { return 0u; }
    // Two BF16 K/V tiles of (head_dim + 8) elements per key row, plus one FP32
    // score-exchange buffer of (kv_tile + 1) floats per (q head, row).
    const uint64_t kv = 2ull * cfg.kv_tile * (cfg.head_dim + 8ull) * 2ull;
    const uint64_t sc = static_cast<uint64_t>(cfg.gqa()) * cfg.q_tile * (cfg.kv_tile + 1ull) * 4ull;
    const uint64_t total = kv + sc;
    return total > 0xFFFFFFFFull ? 0u : static_cast<uint32_t>(total);
}

Status validate_qwen_native_attn_config(const QwenNativeAttnConfig &cfg) noexcept {
    // The device code static_asserts the same three facts; a mismatch here would be
    // an NVRTC failure instead of a Status, so it is checked first.
    if (cfg.head_dim != 256u) {
        return err(StatusCode::unsupported,
                   "native tensor attention is the head_dim 256 model realization",
                   luisa::format("head_dim = {}", cfg.head_dim));
    }
    if (cfg.kv_heads == 0u || cfg.q_heads == 0u || cfg.q_heads % cfg.kv_heads != 0u) {
        return err(StatusCode::unsupported,
                   "native tensor attention GQA requires q_heads % kv_heads == 0",
                   luisa::format("q_heads = {}, kv_heads = {}", cfg.q_heads, cfg.kv_heads));
    }
    if (cfg.gqa() != 4u) {
        return err(StatusCode::unsupported,
                   "native tensor attention packs exactly 4 q heads per kv head (one warp pair each)",
                   luisa::format("q_heads / kv_heads = {}", cfg.gqa()));
    }
    if (cfg.q_tile != 16u) {
        return err(StatusCode::unsupported,
                   "native tensor attention query tile is one m16 fragment (q_tile == 16)",
                   luisa::format("q_tile = {}", cfg.q_tile));
    }
    if (cfg.kv_tile != 32u) {
        return err(StatusCode::unsupported,
                   "native tensor attention key tile is 32 (the 48 KiB static shared-memory ceiling)",
                   luisa::format("kv_tile = {}", cfg.kv_tile));
    }
    if (cfg.block() == 0u || cfg.block() > 1024u) {
        return err(StatusCode::unsupported,
                   "native tensor attention block size must be in [1,1024]",
                   luisa::format("block = {}", cfg.block()));
    }
    const auto bytes = qwen_native_attn_shared_bytes(cfg);
    if (bytes == 0u || bytes > qwen_native_attn_static_smem_limit) {
        return err(StatusCode::unsupported,
                   "native tensor attention tile exceeds the per-block static shared-memory limit",
                   luisa::format("{} B > {} B", bytes, qwen_native_attn_static_smem_limit));
    }
    return Status::success();
}

Status validate_qwen_native_attn_inputs(uint64_t tokens, uint64_t pos0, uint64_t kv_len,
                                        uint64_t capacity, const QwenNativeAttnConfig &cfg) noexcept {
    if (auto s = validate_qwen_native_attn_config(cfg); !s.ok()) { return s; }
    // The causal/capacity contract is the DSL kernels' contract, not a copy of it:
    // pos0 <= kv_len, tokens <= kv_len - pos0 (append first), kv_len <= capacity and
    // the uint32 element envelope.
    if (auto s = validate_qwen_attn_prefill_inputs(tokens, cfg.q_heads, cfg.kv_heads, cfg.head_dim,
                                                   pos0, kv_len, capacity);
        !s.ok()) {
        return s;
    }
    if (capacity > 0xFFFFFFFFull || kv_len > 0xFFFFFFFFull || tokens > 0xFFFFFFFFull ||
        pos0 > 0xFFFFFFFFull) {
        return err(StatusCode::invalid_argument,
                   "native tensor attention geometry exceeds the 32-bit kernel ABI");
    }
    return Status::success();
}

Result<QwenNativeAttnPlan> plan_qwen_native_attn_dispatch(
    uint64_t tokens, const QwenNativeAttnConfig &cfg) noexcept {
    if (auto s = validate_qwen_native_attn_config(cfg); !s.ok()) {
        return Result<QwenNativeAttnPlan>::failure(std::move(s));
    }
    QwenNativeAttnPlan plan;
    plan.block = cfg.block();
    if (tokens == 0u) { return Result<QwenNativeAttnPlan>::success(plan); }
    const auto blocks_x = (tokens + cfg.q_tile - 1u) / cfg.q_tile;
    const auto blocks_y = static_cast<uint64_t>(cfg.kv_heads);
    // The portable 65535-groups-per-dimension envelope the DSL planners enforce is
    // stricter than CUDA's 2^31-1 blocks per dimension and is what a harness may
    // assert against.
    if (blocks_x == 0u || blocks_x > kQwenMaxGroupsPerDim || blocks_y > kQwenMaxGroupsPerDim) {
        return Result<QwenNativeAttnPlan>::failure(err(
            StatusCode::unsupported,
            "native tensor attention grid exceeds the portable per-dimension envelope",
            luisa::format("{},{} blocks", blocks_x, blocks_y)));
    }
    const auto threads_x = blocks_x * plan.block;
    if (threads_x > 0x7FFFFFFFull) {
        return Result<QwenNativeAttnPlan>::failure(err(
            StatusCode::invalid_argument, "native tensor attention dispatch exceeds uint31 threads"));
    }
    plan.blocks_x = static_cast<uint32_t>(blocks_x);
    plan.blocks_y = static_cast<uint32_t>(blocks_y);
    plan.threads_x = static_cast<uint32_t>(threads_x);
    plan.blocks = blocks_x * blocks_y;
    return Result<QwenNativeAttnPlan>::success(plan);
}

luisa::string qwen_native_attn_cuda_source(const QwenNativeAttnConfig &cfg) {
    auto source = luisa::string{qwen_native_attn_cuda_template};
    substitute(source, "@HD@", cfg.head_dim);
    substitute(source, "@QHEADS@", cfg.q_heads);
    substitute(source, "@KVHEADS@", cfg.kv_heads);
    substitute(source, "@GQA@", cfg.gqa());
    substitute(source, "@QTILE@", cfg.q_tile);
    substitute(source, "@KVTILE@", cfg.kv_tile);
    substitute(source, "@WARPS@", cfg.warps());
    substitute(source, "@BLOCK@", cfg.block());
    return source;
}

Status QwenNativeAttn::create(compute::Device &device, const QwenNativeAttnConfig &cfg) noexcept {
    reset();
    if (auto s = validate_qwen_native_attn_config(cfg); !s.ok()) { return s; }
    if (luisa::string_view{device.backend_name()} != "cuda") {
        return err(StatusCode::unsupported,
                   "native tensor attention is CUDA-only (NativeShaderExt CUDA_NVRTC)",
                   luisa::string{device.backend_name()});
    }
    auto *ext = device.extension<NativeShaderExt>();
    if (ext == nullptr) {
        return err(StatusCode::unsupported,
                   "the CUDA backend exposes no NativeShaderExt, so NVRTC native shaders "
                   "are unavailable");
    }
    _ext = ext;
    _cfg = cfg;
    _entry_point = luisa::string{qwen_native_attn_entry};

    luisa::Clock clock;
    clock.tic();
    auto source = qwen_native_attn_cuda_source(cfg);
    NativeShaderCompileInfo info;
    info.language = NativeShaderLanguage::CUDA_NVRTC;
    info.source = luisa::string_view{source};
    info.source_type = NativeShaderSourceType::SourceCode;
    info.entry_point = luisa::string_view{_entry_point};
    info.block_size = make_uint3(cfg.block(), 1u, 1u);
    info.push_constant_size = qwen_native_attn_scalar_bytes;
    info.optimize = true;
    // No fast-math: the FP32 accumulator and the online-softmax statistics must not
    // be reassociated by the compiler, and the only approximations this kernel makes
    // are the explicit ones documented in native_attn_kernel.h.
    info.enable_fast_math = false;
    const auto fail = [&](luisa::string message) -> Status {
        _jit_ms = clock.toc();
        reset();
        return err(StatusCode::internal_error, std::move(message),
                   luisa::format("cold JIT {:.1f} ms", _jit_ms));
    };
    auto result = ext->compile(info);
    if (!result.ok()) {
        _jit_ms = clock.toc();
        reset();
        return err(StatusCode::internal_error,
                   "CUDA NVRTC native tensor attention compilation failed",
                   luisa::string{result.error});
    }
    if (result.bindings.size() != qwen_native_attn_bindings) {
        return fail(luisa::format(
            "native tensor attention reflection reported {} buffer bindings, {} expected",
            result.bindings.size(), qwen_native_attn_bindings));
    }
    auto metadata = ext->load(result);
    if (!metadata.valid()) {
        return fail("native tensor attention load() rejected the compiled module");
    }
    if (metadata.bindings.size() != qwen_native_attn_bindings) {
        ext->destroy_shader(metadata.handle);
        return fail(luisa::format(
            "native tensor attention reflected {} bindings after load, {} expected",
            metadata.bindings.size(), qwen_native_attn_bindings));
    }
    if (any(metadata.block_size != make_uint3(cfg.block(), 1u, 1u))) {
        ext->destroy_shader(metadata.handle);
        return fail(luisa::format("native tensor attention reflected block size ({}, {}, {})",
                                  metadata.block_size.x, metadata.block_size.y,
                                  metadata.block_size.z));
    }
    if (static_cast<bool>(_shader)) { _shader.reset(); }
    _shader = NativeShader{*ext, std::move(metadata)};
    _jit_ms = clock.toc();

    auto bindings = _shader.bindings();
    if (bindings.size() != qwen_native_attn_bindings) {
        reset();
        return err(StatusCode::internal_error,
                   "native tensor attention binding table is not the expected five buffers");
    }
    for (auto i = 0u; i < qwen_native_attn_bindings; i++) { _bindings[i] = bindings[i]; }
    _binding_count = qwen_native_attn_bindings;
    LUISA_INFO("qwen native tensor attention: {} q heads / {} kv heads, head_dim {}, "
               "tile {}x{}, {} warps, {} B static shared, entry '{}', cold JIT {:.1f} ms",
               cfg.q_heads, cfg.kv_heads, cfg.head_dim, cfg.q_tile, cfg.kv_tile,
               cfg.warps(), qwen_native_attn_shared_bytes(cfg), _entry_point, _jit_ms);
    return Status::success();
}

void QwenNativeAttn::reset() noexcept {
    // Destroy the shader instance before the device: `NativeShader::reset()` calls
    // back into the extension that created it.
    _shader.reset();
    _ext = nullptr;
    _binding_count = 0u;
    _entry_point.clear();
}

luisa::unique_ptr<compute::Command> QwenNativeAttn::build_dispatch(
    const compute::BufferView<float> &q,
    const compute::BufferView<float> &gate,
    const compute::BufferView<float> &k_cache,
    const compute::BufferView<float> &v_cache,
    const compute::BufferView<float> &out,
    uint32_t tokens, float scale, uint32_t pos0, uint32_t kv_len, uint32_t capacity,
    luisa::string *error) const noexcept {
    const auto fail = [error](luisa::string message) -> luisa::unique_ptr<compute::Command> {
        if (error != nullptr) { *error = std::move(message); }
        return nullptr;
    };
    if (!valid()) { return fail("native tensor attention is not compiled"); }
    if (!static_cast<bool>(_shader)) { return fail("native tensor attention kernel is not compiled"); }
    auto planned = plan_qwen_native_attn_dispatch(tokens, _cfg);
    if (!planned.ok()) {
        return fail(luisa::format("{}: {}", planned.status.message, planned.status.detail));
    }
    const auto &plan = *planned.value;
    if (plan.empty()) { return fail("native tensor attention has an empty dispatch"); }
    auto launcher = _shader.launcher();
    launcher
        .add_buffer_by_index(0u, q, Usage::READ)
        .add_buffer_by_index(1u, gate, Usage::READ)
        .add_buffer_by_index(2u, k_cache, Usage::READ)
        .add_buffer_by_index(3u, v_cache, Usage::READ)
        .add_buffer_by_index(4u, out, Usage::READ_WRITE)
        .add_uniform(tokens)
        .add_uniform(scale)
        .add_uniform(pos0)
        .add_uniform(kv_len)
        .add_uniform(capacity);
    // The native dispatch takes a THREAD count; the block size (and therefore
    // grid.x) comes from the launcher. grid.y stays the kv head.
    return std::move(launcher).build(make_uint3(plan.threads_x, plan.blocks_y, 1u));
}

}// namespace luisa::qwen::dsl
