// Qwen3.5-4B tensor-core PREFILL GEMM through the CUDA native-shader route
// (opt_plan/09 N1a) -- implementation. See native_gemm.h for the contract and
// native_gemm_kernel.h for the device code.
//
// There is no DSL, no AST and no backend plugin here: the device code is a CUDA
// C++ translation unit that NVRTC compiles at run time (through the CUDA
// backend's NativeShaderExtension) and the launcher binds the five buffers plus
// three scalars by reflection index. That is also why this file only ever
// reports failures: every step (extension lookup, NVRTC, reflection, module
// load) is a hard gate that leaves the object invalid rather than producing a
// half-working accelerator.
#include "qwen/native_gemm.h"

#include <algorithm>
#include <cmath>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/command.h>

#include "qwen/native_gemm_kernel.h"

namespace luisa::qwen::dsl {

namespace {

using compute::NativeShader;
using compute::NativeShaderCompileInfo;
using compute::NativeShaderExt;
using compute::NativeShaderLanguage;
using compute::NativeShaderSourceType;
using compute::Usage;

// Entry point of the generated `__global__` (see native_gemm_kernel.h).
constexpr auto qwen_native_gemm_entry = luisa::string_view{"qwen_native_gemm_bf16"};

// Parameters of the device kernel, in declaration order: five buffer bindings
// followed by three scalar kernel parameters (M, N, K).
constexpr uint32_t qwen_native_gemm_bindings = 5u;
constexpr uint32_t qwen_native_gemm_scalar_bytes = 3u * sizeof(uint32_t);

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

bool qwen_native_gemm_supported_format(QwenWeightFormat format) noexcept {
    // BF16 is the tensor-core native format: the stored codes ARE the mma
    // operands, so the weight path is a pure cp.async copy. FP8 E4M3 gets its own
    // realization: the same `[N,K]` buffer and the same 64-element scale groups,
    // with the codes widened (and scaled) from registers into the BF16 shared tile.
    // Native Q8_0 (4 signed int8 codes per word, one FP32 scale per 32 K elements)
    // is NOT implemented by this kernel: `mma.sync` needs BF16 operands, and
    // widening int8 codes into the BF16 shared tile would need its own measured
    // realization. It is refused here (fail closed), so a dsl-q8 session keeps the
    // DSL MT prefill GEMM for every projection -- correct, and never a silent
    // substitution.
    // Nothing else is accepted, so a future storage format cannot silently reach a
    // kernel that does not implement it.
    return format == QwenWeightFormat::bf16 || format == QwenWeightFormat::fp8_e4m3;
}

Status validate_qwen_native_gemm_config(const QwenNativeGemmConfig &cfg) noexcept {
    if (cfg.block_k != 32u) {
        return err(StatusCode::unsupported,
                   "native tensor GEMM requires a 32-element k step",
                   luisa::format("block_k = {}", cfg.block_k));
    }
    if (cfg.block_m % 32u != 0u || cfg.block_n % 32u != 0u) {
        return err(StatusCode::unsupported,
                   "native tensor GEMM block tile must be a multiple of the 32x32 warp tile",
                   luisa::format("block_m = {}, block_n = {}", cfg.block_m, cfg.block_n));
    }
    if (cfg.warps == 0u || cfg.warps % 2u != 0u) {
        return err(StatusCode::unsupported,
                   "native tensor GEMM needs an even, nonzero warp count",
                   luisa::format("warps = {}", cfg.warps));
    }
    const auto n_warps = cfg.block_n / 32u;
    const auto m_warps = cfg.warps / n_warps;
    if (n_warps == 0u || m_warps == 0u || cfg.warps % n_warps != 0u ||
        cfg.block_m != m_warps * 32u) {
        return err(StatusCode::unsupported,
                   "native tensor GEMM warp grid must tile the block tile exactly",
                   luisa::format("block_m = {}, block_n = {}, warps = {}",
                                 cfg.block_m, cfg.block_n, cfg.warps));
    }
    // Stage A holds 16 FP32 activations per thread; the register prefetch of the
    // next stage is sized by that (the device code static_asserts it too).
    if (static_cast<uint64_t>(cfg.block_m) * cfg.block_k != 16ull * cfg.block()) {
        return err(StatusCode::unsupported,
                   "native tensor GEMM stage A must be 16 floats per thread",
                   luisa::format("block_m = {}, block_k = {}, block = {}",
                                 cfg.block_m, cfg.block_k, cfg.block()));
    }
    return Status::success();
}

Status validate_qwen_native_gemm_shape(uint64_t m, uint64_t n, uint64_t k,
                                       QwenWeightFormat format, uint32_t fp8_group) noexcept {
    if (!qwen_native_gemm_supported_format(format)) {
        return err(StatusCode::unsupported,
                   "native tensor GEMM has no realization for this weight format",
                   luisa::string{qwen_weight_format_name(format)});
    }
    if (k == 0u || (k & 1u) != 0u) {
        return err(StatusCode::unsupported,
                   "native tensor GEMM needs a nonzero even K (packed-word ABI)",
                   luisa::format("k = {}", k));
    }
    if (format == QwenWeightFormat::fp8_e4m3) {
        // One packed word is 4 codes, and the widening reads a whole word: K must
        // be a multiple of 4 so no word straddles a row end, and the scale group
        // must be a multiple of 4 so no word straddles a group (one scale load per
        // word). Both hold for the model (K = 2560/4096/9216, group 64).
        if ((k & 3u) != 0u) {
            return err(StatusCode::unsupported,
                       "native FP8 tensor GEMM needs K % 4 == 0 (a packed word must not "
                       "straddle a row end)",
                       luisa::format("k = {}", k));
        }
        if (fp8_group == 0u || (fp8_group & 3u) != 0u) {
            return err(StatusCode::unsupported,
                       "native FP8 tensor GEMM needs a scale group that is a nonzero multiple "
                       "of 4",
                       luisa::format("group = {}", fp8_group));
        }
    }
    if (n == 0u) {
        return err(StatusCode::invalid_argument, "native tensor GEMM needs n > 0");
    }
    if (m == 0u) {
        return Status::success();// nothing to dispatch
    }
    if (m > 0xFFFFFFFFull || n > 0xFFFFFFFFull || k > 0xFFFFFFFFull) {
        return err(StatusCode::invalid_argument,
                   "native tensor GEMM geometry exceeds the 32-bit kernel ABI");
    }
    return Status::success();
}

Result<QwenNativeGemmPlan> plan_qwen_native_gemm_dispatch(
    uint64_t m, uint64_t n, uint64_t k, const QwenNativeGemmConfig &cfg) noexcept {
    if (auto s = validate_qwen_native_gemm_config(cfg); !s.ok()) {
        return Result<QwenNativeGemmPlan>::failure(std::move(s));
    }
    QwenNativeGemmPlan plan;
    plan.block = cfg.block();
    if (m == 0u || n == 0u || k == 0u) { return Result<QwenNativeGemmPlan>::success(plan); }
    const auto blocks_x = (m + cfg.block_m - 1u) / cfg.block_m;
    const auto blocks_y = (n + cfg.block_n - 1u) / cfg.block_n;
    // The CUDA grid envelope is 2^31-1 blocks per dimension; the portable
    // 65535-groups-per-dimension rule the DSL planners enforce is stricter and
    // is what a benchmark harness may assert against.
    if (blocks_x == 0u || blocks_y == 0u ||
        blocks_x > kQwenMaxGroupsPerDim || blocks_y > kQwenMaxGroupsPerDim) {
        return Result<QwenNativeGemmPlan>::failure(err(
            StatusCode::unsupported,
            "native tensor GEMM grid exceeds the portable per-dimension envelope",
            luisa::format("{},{} blocks", blocks_x, blocks_y)));
    }
    const auto threads_x = static_cast<uint64_t>(blocks_x) * plan.block;
    if (threads_x > 0x7FFFFFFFull) {
        return Result<QwenNativeGemmPlan>::failure(err(
            StatusCode::invalid_argument, "native tensor GEMM dispatch exceeds uint31 threads"));
    }
    plan.blocks_x = static_cast<uint32_t>(blocks_x);
    plan.blocks_y = static_cast<uint32_t>(blocks_y);
    plan.threads_x = static_cast<uint32_t>(threads_x);
    plan.blocks = blocks_x * blocks_y;
    return Result<QwenNativeGemmPlan>::success(plan);
}

luisa::string qwen_native_gemm_cuda_source(const QwenNativeGemmConfig &cfg, bool residual,
                                           QwenWeightFormat format) {
    auto source = luisa::string{qwen_native_gemm_cuda_template};
    substitute(source, "@BM@", cfg.block_m);
    substitute(source, "@BN@", cfg.block_n);
    substitute(source, "@BK@", cfg.block_k);
    substitute(source, "@WARPS@", cfg.warps);
    substitute(source, "@BLOCK@", cfg.block());
    substitute(source, "@RESIDUAL@", residual ? 1u : 0u);
    substitute(source, "@FP8@", format == QwenWeightFormat::fp8_e4m3 ? 1u : 0u);
    substitute(source, "@GROUP@", cfg.fp8_group);
    return source;
}

Status QwenNativeGemm::create(compute::Device &device, QwenWeightFormat format,
                              const QwenNativeGemmConfig &cfg) noexcept {
    reset();
    if (!qwen_native_gemm_supported_format(format)) {
        return err(StatusCode::unsupported,
                   "native tensor GEMM has no realization for this weight format",
                   luisa::string{qwen_weight_format_name(format)});
    }
    if (auto s = validate_qwen_native_gemm_config(cfg); !s.ok()) { return s; }
    // The shape gate is per-format, and for FP8 it also pins the scale group the
    // generated source bakes in.
    // A minimal probe that only the format/group rules can reject (K = 4 satisfies
    // every geometry rule of both realizations).
    if (auto s = validate_qwen_native_gemm_shape(1u, 1u, 4u, format, cfg.fp8_group); !s.ok()) {
        return s;
    }
    if (luisa::string_view{device.backend_name()} != "cuda") {
        return err(StatusCode::unsupported,
                   "native tensor GEMM is CUDA-only (NativeShaderExt CUDA_NVRTC)",
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
    _format = format;
    _fp8 = format == QwenWeightFormat::fp8_e4m3;
    _entry_point = luisa::string{qwen_native_gemm_entry};

    luisa::Clock clock;
    clock.tic();
    const auto compile_one = [&](bool residual, NativeShader &out,
                                 luisa::string &error) -> bool {
        auto source = qwen_native_gemm_cuda_source(cfg, residual, format);
        NativeShaderCompileInfo info;
        info.language = NativeShaderLanguage::CUDA_NVRTC;
        info.source = luisa::string_view{source};
        info.source_type = NativeShaderSourceType::SourceCode;
        info.entry_point = luisa::string_view{_entry_point};
        info.block_size = make_uint3(cfg.block(), 1u, 1u);
        info.push_constant_size = qwen_native_gemm_scalar_bytes;
        info.optimize = true;
        // No fast-math: the FP32 accumulator must not be reassociated by the
        // compiler, and the only transcendental-free arithmetic here is exact.
        info.enable_fast_math = false;
        auto result = ext->compile(info);
        if (!result.ok()) {
            error = luisa::string{result.error};
            return false;
        }
        if (result.bindings.size() != qwen_native_gemm_bindings) {
            error = luisa::format(
                "native tensor GEMM reflection reported {} buffer bindings, {} expected",
                result.bindings.size(), qwen_native_gemm_bindings);
            return false;
        }
        auto metadata = ext->load(result);
        if (!metadata.valid()) {
            error = "native tensor GEMM load() rejected the compiled module";
            return false;
        }
        if (metadata.bindings.size() != qwen_native_gemm_bindings) {
            error = luisa::format(
                "native tensor GEMM reflected {} bindings after load, {} expected",
                metadata.bindings.size(), qwen_native_gemm_bindings);
            ext->destroy_shader(metadata.handle);
            return false;
        }
        if (any(metadata.block_size != make_uint3(cfg.block(), 1u, 1u))) {
            error = luisa::format("native tensor GEMM reflected block size ({}, {}, {})",
                                  metadata.block_size.x, metadata.block_size.y,
                                  metadata.block_size.z);
            ext->destroy_shader(metadata.handle);
            return false;
        }
        if (static_cast<bool>(out)) { out.reset(); }
        out = NativeShader{*ext, std::move(metadata)};
        return true;
    };

    luisa::string error;
    if (!compile_one(false, _plain, error)) {
        _jit_ms = clock.toc();
        reset();
        return err(StatusCode::internal_error,
                   "CUDA NVRTC native tensor GEMM compilation failed", std::move(error));
    }
    if (!compile_one(true, _residual, error)) {
        _jit_ms = clock.toc();
        reset();
        return err(StatusCode::internal_error,
                   "CUDA NVRTC native tensor GEMM (residual) compilation failed",
                   std::move(error));
    }
    _jit_ms = clock.toc();

    auto bindings = _plain.bindings();
    if (bindings.size() != qwen_native_gemm_bindings) {
        reset();
        return err(StatusCode::internal_error,
                   "native tensor GEMM binding table is not the expected five buffers");
    }
    for (auto i = 0u; i < qwen_native_gemm_bindings; i++) { _bindings[i] = bindings[i]; }
    _binding_count = qwen_native_gemm_bindings;
    LUISA_INFO("qwen native tensor GEMM: {} tile {}x{}x{} ({} warps), entry '{}', "
               "cold JIT {:.1f} ms",
               qwen_weight_format_name(format), cfg.block_m, cfg.block_n, cfg.block_k,
               cfg.warps, _entry_point, _jit_ms);
    return Status::success();
}

void QwenNativeGemm::reset() noexcept {
    // Destroy the shader instances before the device: `NativeShader::reset()`
    // calls back into the extension that created them.
    _residual.reset();
    _plain.reset();
    _ext = nullptr;
    _binding_count = 0u;
    _entry_point.clear();
}

luisa::unique_ptr<compute::Command> QwenNativeGemm::build_dispatch(
    bool residual_epilogue,
    const compute::BufferView<float> &x,
    const compute::BufferView<uint32_t> &w,
    const compute::BufferView<float> &scales,
    const compute::BufferView<float> &residual,
    const compute::BufferView<float> &y,
    uint32_t m, uint32_t n, uint32_t k,
    luisa::string *error) const noexcept {
    const auto fail = [error](luisa::string message) -> luisa::unique_ptr<compute::Command> {
        if (error != nullptr) { *error = std::move(message); }
        return nullptr;
    };
    if (!valid()) { return fail("native tensor GEMM is not compiled"); }
    const auto *shader = residual_epilogue ? &_residual : &_plain;
    if (!static_cast<bool>(*shader)) { return fail("native tensor GEMM variant is not compiled"); }
    auto planned = plan_qwen_native_gemm_dispatch(m, n, k, _cfg);
    if (!planned.ok()) {
        return fail(luisa::format("{}: {}", planned.status.message, planned.status.detail));
    }
    const auto &plan = *planned.value;
    if (plan.empty()) { return fail("native tensor GEMM has an empty dispatch"); }
    auto launcher = shader->launcher();
    launcher
        .add_buffer_by_index(0u, x, Usage::READ)
        .add_buffer_by_index(1u, w, Usage::READ)
        .add_buffer_by_index(2u, scales, Usage::READ)
        .add_buffer_by_index(3u, residual, Usage::READ)
        .add_buffer_by_index(4u, y, Usage::READ_WRITE)
        .add_uniform(m)
        .add_uniform(n)
        .add_uniform(k);
    // The native dispatch takes a THREAD count; the block size (and therefore
    // grid.x) comes from the launcher. grid.y stays the N tile count.
    return std::move(launcher).build(make_uint3(plan.threads_x, plan.blocks_y, 1u));
}

}// namespace luisa::qwen::dsl
