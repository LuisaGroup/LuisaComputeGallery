// Qwen3.5-4B PREFILL GEMM on the CUDA native Tile IR route, typed FP8 weight
// storage -- implementation. See cuda_tile_gemm.h for the contract, the numerics
// and the fail-closed rules.
//
// There is no host repack here: the kernel is captured from the Tile DSL for one
// static (m, n, k), compiled with `tile::compile` (Lowering::NATIVE) and invoked
// with the caller's OWN buffer views. The packed uint32 FP8 weight buffer is
// reinterpreted as `tile::float8_e4m3fn` element-wise, which is a view type
// change only: identical handle, offset and bytes.
#include "qwen/cuda_tile_gemm.h"

#include <luisa/core/clock.h>
#include <luisa/core/platform.h>
#include <luisa/core/stl/format.h>
#include <luisa/tile/runtime.h>

namespace luisa::qwen::dsl {

namespace {

[[nodiscard]] Status err(StatusCode code, luisa::string msg, luisa::string detail = {}) {
    return Status::error(code, std::move(msg), std::move(detail));
}

[[nodiscard]] bool power_of_two(uint32_t value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

// The two kernels differ ONLY in the epilogue: the residual load/store is
// captured out of the plain variant so the plain shader has no residual argument
// use at all.
[[nodiscard]] constexpr luisa::string_view kernel_name(bool residual_epilogue) noexcept {
    return residual_epilogue ? luisa::string_view{"qwen_cuda_tile_fp8_gemm_residual"} :
                               luisa::string_view{"qwen_cuda_tile_fp8_gemm"};
}

[[nodiscard]] QwenCudaTileGemmCapture finish_capture(ct::Kernel kernel) noexcept {
    QwenCudaTileGemmCapture capture;
    for (auto &&diagnostic : kernel.diagnostics()) { capture.diagnostics.emplace_back(diagnostic); }
    if (!kernel.valid()) {
        auto verification = ct::verify(kernel.module());
        for (auto &&diagnostic : verification.diagnostics()) {
            capture.diagnostics.emplace_back(diagnostic.message);
        }
        capture.status = err(StatusCode::internal_error,
                             "native Tile FP8 GEMM capture failed verification",
                             capture.diagnostics.empty() ?
                                 luisa::string{"no diagnostics"} :
                                 capture.diagnostics.front());
        capture.kernel.emplace(std::move(kernel));
        return capture;
    }
    capture.kernel.emplace(std::move(kernel));
    return capture;
}

// The two compile probes: load E4M3 codes, widen them to BF16, widen that to FP32
// and store. They exercise exactly the constructs this module needs from the
// native Tile path (an FP8 buffer parameter, an FP8 tile load, the E4M3 -> BF16
// cast, and - in the ragged variant - a PARTIAL FP8 tile load) and nothing else,
// so if they compile the rest of the GEMM is ordinary BF16 MMA and FP32
// arithmetic. Their purpose is to reject a CUDA < 13.4 toolkit, a missing
// tileiras, or a target the native backend cannot serve at create() time instead
// of on the first prompt.
[[nodiscard]] QwenCudaTileGemmCapture capture_storage_probe(bool ragged) noexcept {
    // `ragged = false`: one program, one full 64-element tile, an exact in-bounds
    // typed FP8 load. `ragged = true`: two programs over a 96-element view, so the
    // second one reads a PARTIAL FP8 tile and the backend has to emit a masked
    // FP8 load (the construct every ragged N needs).
    constexpr auto extent = qwen_cuda_tile_gemm_k_step * 3u / 2u;// 96: one full tile + one tail
    auto definition = ct::tile_kernel(
        ragged ? "qwen_cuda_tile_fp8_ragged_probe" : "qwen_cuda_tile_fp8_storage_probe",
        [ragged](ct::TensorView<const ct::float8_e4m3fn, 1> src, ct::TensorView<float, 1> dst) {
            auto i = ct::axis("i", qwen_cuda_tile_gemm_k_step);
            auto b = ct::axis("b", ragged ? 2 : 1);
            for (auto &g : ct::parallel(ct::shape(b))) {
                auto origin = ct::coord(g.index(b) * static_cast<int64_t>(qwen_cuda_tile_gemm_k_step));
                auto codes = src.tile(origin, ct::shape(i)).load();
                dst(origin, ct::shape(i)).store(ct::cast<float>(ct::cast<ct::bfloat16>(codes)));
            }
        });
    auto view_extent = ragged ? extent : qwen_cuda_tile_gemm_k_step;
    return finish_capture(definition.capture(ct::tensor_shape(view_extent), ct::tensor_shape(view_extent)));
}

}// namespace

bool qwen_cuda_tile_gemm_supported_format(QwenWeightFormat format) noexcept {
    // Only the typed E4M3 storage exists on this route. BF16 would need the
    // `bfloat16` element (already a scalar there, but the kernel's scale-after-dot
    // contract is defined for the FP8 packing), and Q8_0/W4 are packed word
    // formats whose software decode is explicitly NOT implemented natively:
    // those keep the DSL MT / native-shader path.
    return format == QwenWeightFormat::fp8_e4m3;
}

bool qwen_cuda_tile_gemm_native_route_requested() noexcept {
    auto value = luisa::get_environment_variable("LUISA_CUDA_TILE_IR");
    return value && luisa::string_view{*value} == "1";
}

Status validate_qwen_cuda_tile_gemm_config(const QwenCudaTileGemmConfig &cfg) noexcept {
    if (cfg.tile_k != qwen_cuda_tile_gemm_k_step) {
        return err(StatusCode::unsupported,
                   "native Tile FP8 GEMM requires a 64-element k step (the scale group)",
                   luisa::format("tile_k = {}", cfg.tile_k));
    }
    if (cfg.group != qwen_cuda_tile_gemm_group) {
        return err(StatusCode::unsupported,
                   "native Tile FP8 GEMM requires the 64-element FP8 scale group",
                   luisa::format("group = {}", cfg.group));
    }
    if (!power_of_two(cfg.tile_m) || !power_of_two(cfg.tile_n)) {
        return err(StatusCode::invalid_argument,
                   "native Tile FP8 GEMM tile_m/tile_n must be powers of two",
                   luisa::format("tile_m = {}, tile_n = {}", cfg.tile_m, cfg.tile_n));
    }
    if (cfg.tile_m < qwen_cuda_tile_gemm_min_tile || cfg.tile_m > qwen_cuda_tile_gemm_max_tile ||
        cfg.tile_n < qwen_cuda_tile_gemm_min_tile || cfg.tile_n > qwen_cuda_tile_gemm_max_tile) {
        return err(StatusCode::invalid_argument,
                   "native Tile FP8 GEMM tile_m/tile_n out of range",
                   luisa::format("tile_m = {}, tile_n = {}, allowed [{}, {}]", cfg.tile_m, cfg.tile_n,
                                 qwen_cuda_tile_gemm_min_tile, qwen_cuda_tile_gemm_max_tile));
    }
    if (cfg.tile_m * cfg.tile_n > qwen_cuda_tile_gemm_max_tile_elements ||
        cfg.tile_m * cfg.tile_k > qwen_cuda_tile_gemm_max_tile_elements ||
        cfg.tile_n * cfg.tile_k > qwen_cuda_tile_gemm_max_tile_elements) {
        return err(StatusCode::invalid_argument,
                   "native Tile FP8 GEMM tile volume exceeds the native Tile budget",
                   luisa::format("tile_m = {}, tile_n = {}, tile_k = {}, budget {} elements",
                                 cfg.tile_m, cfg.tile_n, cfg.tile_k,
                                 qwen_cuda_tile_gemm_max_tile_elements));
    }
    return Status::success();
}

Status validate_qwen_cuda_tile_gemm_shape(uint64_t m, uint64_t n, uint64_t k,
                                          QwenWeightFormat format, uint32_t group) noexcept {
    if (!qwen_cuda_tile_gemm_supported_format(format)) {
        return err(StatusCode::unsupported,
                   "native Tile FP8 GEMM supports only fp8_e4m3 weight storage",
                   luisa::string{qwen_weight_format_name(format)});
    }
    if (group != qwen_cuda_tile_gemm_group) {
        return err(StatusCode::unsupported,
                   "native Tile FP8 GEMM requires the 64-element FP8 scale group",
                   luisa::format("group = {}", group));
    }
    // A static Tile capture needs positive extents: there is no "empty grid" Tile
    // kernel, so a zero extent is refused instead of being silently dropped.
    if (m == 0u || n == 0u || k == 0u) {
        return err(StatusCode::invalid_argument,
                   "native Tile FP8 GEMM shape must have nonzero m, n and k",
                   luisa::format("m = {}, n = {}, k = {}", m, n, k));
    }
    if (k % qwen_cuda_tile_gemm_k_step != 0u) {
        return err(StatusCode::unsupported,
                   "native Tile FP8 GEMM requires K to be a multiple of the 64-element k step (no K tail)",
                   luisa::format("k = {} (remainder {})", k, k % qwen_cuda_tile_gemm_k_step));
    }
    if (m > qwen_cuda_tile_gemm_max_extent || n > qwen_cuda_tile_gemm_max_extent ||
        k > qwen_cuda_tile_gemm_max_extent) {
        return err(StatusCode::invalid_argument,
                   "native Tile FP8 GEMM extent exceeds the static bound",
                   luisa::format("m = {}, n = {}, k = {}, bound {}", m, n, k,
                                 qwen_cuda_tile_gemm_max_extent));
    }
    const auto volume_ok = [](uint64_t a, uint64_t b, const char *what, luisa::string &detail) noexcept {
        if (b != 0u && a > qwen_cuda_tile_gemm_max_elements / b) {
            detail = luisa::format("{} = {} x {} exceeds {} elements", what, a, b,
                                   qwen_cuda_tile_gemm_max_elements);
            return false;
        }
        return true;
    };
    luisa::string detail;
    if (!volume_ok(m, k, "m*k", detail) || !volume_ok(n, k, "n*k", detail) ||
        !volume_ok(m, n, "m*n", detail)) {
        return err(StatusCode::invalid_argument, "native Tile FP8 GEMM volume overflow", std::move(detail));
    }
    // Root grid: one program per (tile_m, tile_n) output tile. The captured
    // tile_m/tile_n come from the config, so use the smallest allowed tile as the
    // conservative divisor here; build_dispatch re-checks with the real tiling.
    const auto programs = ((m + qwen_cuda_tile_gemm_min_tile - 1u) / qwen_cuda_tile_gemm_min_tile) *
                          ((n + qwen_cuda_tile_gemm_min_tile - 1u) / qwen_cuda_tile_gemm_min_tile);
    if (programs > 0x7fffffffull) {
        return err(StatusCode::invalid_argument, "native Tile FP8 GEMM program grid exceeds the CUDA grid limit",
                   luisa::format("programs = {}", programs));
    }
    return Status::success();
}

QwenCudaTileGemmCapture capture_qwen_cuda_tile_gemm(const QwenCudaTileGemmConfig &cfg, uint32_t m,
                                                    uint32_t n, uint32_t k, bool residual_epilogue) noexcept {
    QwenCudaTileGemmCapture capture;
    if (auto status = validate_qwen_cuda_tile_gemm_config(cfg); !status.ok()) {
        capture.status = std::move(status);
        return capture;
    }
    if (auto status = validate_qwen_cuda_tile_gemm_shape(m, n, k, QwenWeightFormat::fp8_e4m3, cfg.group);
        !status.ok()) {
        capture.status = std::move(status);
        return capture;
    }
    const auto M = static_cast<int64_t>(m);
    const auto N = static_cast<int64_t>(n);
    const auto K = static_cast<int64_t>(k);
    const auto TM = static_cast<int64_t>(cfg.tile_m);
    const auto TN = static_cast<int64_t>(cfg.tile_n);
    const auto BK = static_cast<int64_t>(cfg.tile_k);
    const auto k_blocks = K / BK;
    const auto groups = K / static_cast<int64_t>(cfg.group);// == k_blocks by construction
    auto definition = ct::tile_kernel(
        kernel_name(residual_epilogue),
        [=](ct::TensorView<const float, 2> x,
            ct::TensorView<const ct::float8_e4m3fn, 2> w,
            ct::TensorView<const float, 2> scales,
            ct::TensorView<const float, 2> residual,
            ct::TensorView<float, 2> y) {
            auto m_axis = ct::axis("m", TM);
            auto n_axis = ct::axis("n", TN);
            auto k_axis = ct::axis("k", BK);
            auto gm = ct::axis("gm", (M + TM - 1) / TM);
            auto gn = ct::axis("gn", (N + TN - 1) / TN);
            // Space of ONE scale column: positionally [n, 1] so its addresses are
            // the stride-`groups` column of the [N, K/64] scale tensor, but its
            // singleton axis IS `m_axis` (extent 1) rather than a private axis.
            // That makes `partial[m,n] * scale[n,m=1]` a plain named-axis
            // broadcast (permute + broadcast on the native route) instead of a
            // squeeze/gather, which the native CUDA Tile realization does not
            // implement. The values loaded are exactly scale[n0 + i, block].
            ct::IndexSpace scale_space;
            static_cast<void>(scale_space.add(n_axis.dimension(), static_cast<uint64_t>(TN)));
            static_cast<void>(scale_space.add(m_axis.dimension(), static_cast<uint64_t>(1u)));
            // One program per output tile; ragged M/N simply fall out of the
            // default bounds::zero masking of the loads and the store.
            for (auto &program : ct::parallel(ct::shape(gm, gn))) {
                auto m0 = program.index(gm) * TM;
                auto n0 = program.index(gn) * TN;
                auto acc = ct::zeros<float>(ct::shape(m_axis, n_axis));
                for (auto &block : ct::serial(ct::shape(k_blocks))) {
                    auto k0 = block.index() * BK;
                    auto a = x.tile(ct::coord(m0, k0), ct::shape(m_axis, k_axis)).load();
                    auto b = w.tile(ct::coord(n0, k0), ct::shape(n_axis, k_axis)).load();
                    auto a_bf16 = ct::cast<ct::bfloat16>(a);
                    auto b_bf16 = ct::cast<ct::bfloat16>(b);
                    // BF16 tensor-core partial dot with an FP32 accumulator. The
                    // E4M3 -> BF16 widening of the weights is exact, so the only
                    // rounding in this contraction is the activation's.
                    auto partial = ct::mma(a_bf16, b_bf16, ct::zeros<float>(ct::shape(m_axis, n_axis)),
                                           ct::MmaPolicy{.allow_reassociation = true});
                    // The FP32 group scale multiplies the PARTIAL DOT (not the
                    // individual weights): scale AFTER the contraction, always in
                    // FP32, so no narrow arithmetic and no scale-rounding of the
                    // weight operand is involved.
                    auto scale = scales.tile(ct::coord(n0, block.index()), scale_space).load();
                    acc = acc + partial * scale;
                }
                if (residual_epilogue) {
                    acc = acc + residual.tile(ct::coord(m0, n0), ct::shape(m_axis, n_axis)).load();
                }
                y(ct::coord(m0, n0), ct::shape(m_axis, n_axis)).store(acc);
            }
        });
    // The packed uint32 buffer holds K/4 words per output row; the FP8 view is
    // [N, K] elements and each row starts on a word boundary because K % 64 == 0.
    auto dummy = ct::tensor_shape(1, 1);
    auto residual_shape = residual_epilogue ? ct::tensor_shape(M, N) : dummy;
    return finish_capture(definition.capture(ct::tensor_shape(M, K), ct::tensor_shape(N, K),
                                             ct::tensor_shape(N, groups), residual_shape,
                                             ct::tensor_shape(M, N)));
}

uint64_t QwenCudaTileGemm::_key(uint32_t m, uint32_t n, uint32_t k, bool residual) noexcept {
    // 21 bits per extent (validate_qwen_cuda_tile_gemm_shape bounds them to 2^20)
    // plus the epilogue bit; nothing else is part of the specialization.
    return (static_cast<uint64_t>(residual) << 63u) |
           (static_cast<uint64_t>(m) << 42u) |
           (static_cast<uint64_t>(n) << 21u) |
           static_cast<uint64_t>(k);
}

Status QwenCudaTileGemm::_probe(compute::Device &device) noexcept {
    // Two construct-specific probes, tried in this order:
    //  1. a RAGGED typed-FP8 load (two programs over a 96-element view) - it
    //     subsumes the exact-tile construct and additionally needs a masked FP8
    //     tile load, which the native backend only supports for SM90+;
    //  2. an exact-tile typed-FP8 load, the minimum this module needs.
    // The result decides `_ragged_fp8_supported`: probing instead of guessing an
    // SM number keeps the gate honest on toolkits that change the restriction.
    struct ProbeResult {
        bool compiled{false};
        luisa::string detail;
    };
    auto run = [&device, this](bool ragged) noexcept {
        ProbeResult result;
        auto capture = capture_storage_probe(ragged);
        if (!capture.ok()) {
            result.detail = capture.status.message;
            if (!capture.status.detail.empty()) {
                result.detail.append(": ");
                result.detail.append(capture.status.detail);
            }
            for (auto &&diagnostic : capture.diagnostics) {
                result.detail.append("\n  diag: ");
                result.detail.append(diagnostic);
            }
            return result;
        }
        ct::CompileOptions options;            // NATIVE lowering, planner default
        compute::ShaderOption shader_option{}; // enable_fast_math = false
        shader_option.enable_fast_math = false;
        luisa::Clock clock;
        auto shader = ct::compile(device, *capture.kernel, options, shader_option);
        _probe_ms += clock.toc();
        if (!shader) {
            result.detail = shader.metadata().error;
            for (auto &&diagnostic : capture.diagnostics) {
                result.detail.append("\n  diag: ");
                result.detail.append(diagnostic);
            }
            return result;
        }
        result.compiled = true;
        return result;
    };
    auto ragged = run(true);
    _ragged_fp8_supported = ragged.compiled;
    if (ragged.compiled) { return Status::success(); }
    auto exact = run(false);
    if (!exact.compiled) {
        luisa::string detail = "exact-tile probe: ";
        detail.append(exact.detail);
        detail.append("\nragged-tile probe: ");
        detail.append(ragged.detail);
        return err(StatusCode::unsupported,
                   "CUDA native Tile IR cannot compile typed FP8 storage on this device/toolkit "
                   "(CUDA 13.4+ with tileiras and a supported SM are required; see the "
                   "luisa-cuda-tile-fp8-storage.patch note in AGENTS.md)",
                   std::move(detail));
    }
    return Status::success();
}

Status QwenCudaTileGemm::create(compute::Device &device, const QwenCudaTileGemmConfig &cfg) noexcept {
    reset();
    if (auto status = validate_qwen_cuda_tile_gemm_config(cfg); !status.ok()) { return status; }
    auto backend = luisa::string_view{device.backend_name()};
    if (backend != "cuda") {
        return err(StatusCode::unsupported,
                   "native Tile FP8 GEMM requires the cuda backend (dx/vk reject FP8 Tile storage)",
                   luisa::string{backend});
    }
    if (!qwen_cuda_tile_gemm_native_route_requested()) {
        return err(StatusCode::unsupported,
                   "CUDA native Tile IR is not requested: set LUISA_CUDA_TILE_IR=1",
                   "the CUDA backend requires an explicit opt-in for its experimental native Tile "
                   "IR route; this module never substitutes another lowering or the "
                   "DSL/native-shader GEMM");
    }
    _probe_ms = 0.0;
    _ragged_fp8_supported = false;
    if (auto status = _probe(device); !status.ok()) { return status; }
    _device = &device;
    _cfg = cfg;
    _compile_count = 0u;
    _cache_hits = 0u;
    _jit_ms = 0.0;
    return Status::success();
}

void QwenCudaTileGemm::reset() noexcept {
    _shaders.clear();
    _device = nullptr;
    _cfg = QwenCudaTileGemmConfig{};
    _compile_count = 0u;
    _cache_hits = 0u;
    _jit_ms = 0.0;
    _probe_ms = 0.0;
    _ragged_fp8_supported = false;
}

ct::Shader *QwenCudaTileGemm::_shader_for(uint32_t m, uint32_t n, uint32_t k, bool residual,
                                          luisa::string &error) noexcept {
    auto key = _key(m, n, k, residual);
    if (auto it = _shaders.find(key); it != _shaders.end()) {
        _cache_hits++;
        return it->second.get();
    }
    if (_device == nullptr) {
        error = "native Tile FP8 GEMM is not valid (create() failed or was not called)";
        return nullptr;
    }
    auto capture = capture_qwen_cuda_tile_gemm(_cfg, m, n, k, residual);
    if (!capture.ok()) {
        error = luisa::format("{}: {}", capture.status.message, capture.status.detail);
        for (auto &&diagnostic : capture.diagnostics) {
            error.append("\n  diag: ");
            error.append(diagnostic);
        }
        return nullptr;
    }
    ct::CompileOptions options;            // NATIVE lowering only: no TIRX substitution
    compute::ShaderOption shader_option{};
    shader_option.enable_fast_math = false;// FP32 fidelity for the scale epilogue
    luisa::Clock clock;
    auto shader = ct::compile(*_device, *capture.kernel, options, shader_option);
    _jit_ms += clock.toc();
    if (!shader) {
        error = shader.metadata().error;
        for (auto &&diagnostic : capture.diagnostics) {
            error.append("\n  diag: ");
            error.append(diagnostic);
        }
        return nullptr;
    }
    _compile_count++;
    auto owned = luisa::make_unique<ct::Shader>(std::move(shader));
    auto *raw = owned.get();
    _shaders.emplace(key, std::move(owned));
    return raw;
}

luisa::unique_ptr<compute::Command> QwenCudaTileGemm::build_dispatch(
    bool residual_epilogue,
    const compute::BufferView<float> &x,
    const compute::BufferView<uint32_t> &w,
    const compute::BufferView<float> &scales,
    const compute::BufferView<float> &residual,
    const compute::BufferView<float> &y,
    uint32_t m, uint32_t n, uint32_t k,
    luisa::string *error) noexcept {
    const auto fail = [error](StatusCode code, luisa::string message, luisa::string detail = {}) noexcept {
        if (error != nullptr) {
            *error = message;
            if (!detail.empty()) {
                error->append(": ");
                error->append(detail);
            }
        }
        static_cast<void>(code);
        return luisa::unique_ptr<compute::Command>{};
    };
    if (_device == nullptr) {
        return fail(StatusCode::unsupported,
                    "native Tile FP8 GEMM is not valid (create() failed or was not called)");
    }
    if (auto status = validate_qwen_cuda_tile_gemm_shape(m, n, k, QwenWeightFormat::fp8_e4m3, _cfg.group);
        !status.ok()) {
        return fail(status.code, status.message, status.detail);
    }
    if (!_ragged_fp8_supported && uint64_t(n) % _cfg.tile_n != 0u) {
        return fail(StatusCode::unsupported,
                    "native Tile FP8 GEMM refuses a ragged N on this target",
                    luisa::format("n = {} is not a multiple of tile_n = {} and this device/toolkit "
                                  "cannot compile a PARTIAL (masked) FP8 tile load; pick a tile_n "
                                  "that divides n or keep the DSL/native-shader GEMM for this "
                                  "projection (M may always be ragged: its tiles are FP32)",
                                  n, _cfg.tile_n));
    }
    const auto programs_x = (uint64_t(m) + _cfg.tile_m - 1u) / _cfg.tile_m;
    const auto programs_y = (uint64_t(n) + _cfg.tile_n - 1u) / _cfg.tile_n;
    if (programs_x * programs_y > 0x7fffffffull) {
        return fail(StatusCode::invalid_argument, "native Tile FP8 GEMM program grid exceeds the CUDA grid limit",
                    luisa::format("{} x {} programs", programs_x, programs_y));
    }
    // Every view is checked against the captured static shape BEFORE a command is
    // built: nothing here relies on the runtime argument assertions for user input.
    const auto need_x = uint64_t(m) * k;
    const auto need_words = (uint64_t(n) * k + 3u) / 4u;
    const auto need_scales = uint64_t(n) * (k / _cfg.group);
    const auto need_y = uint64_t(m) * n;
    if (uint64_t(x.size()) < need_x) {
        return fail(StatusCode::invalid_argument, "native Tile FP8 GEMM activation view is too small",
                    luisa::format("x.size {} < m*k {}", x.size(), need_x));
    }
    if (uint64_t(w.size()) < need_words) {
        return fail(StatusCode::invalid_argument, "native Tile FP8 GEMM weight view is too small",
                    luisa::format("w.size {} < n*k/4 {}", w.size(), need_words));
    }
    if (uint64_t(scales.size()) < need_scales) {
        return fail(StatusCode::invalid_argument, "native Tile FP8 GEMM scale view is too small",
                    luisa::format("scales.size {} < n*(k/64) {}", scales.size(), need_scales));
    }
    if (residual_epilogue && uint64_t(residual.size()) < need_y) {
        return fail(StatusCode::invalid_argument, "native Tile FP8 GEMM residual view is too small",
                    luisa::format("residual.size {} < m*n {}", residual.size(), need_y));
    }
    if (uint64_t(residual.size()) < 1u) {
        // The plain capture still binds a 1-element residual (the dummy ABI slot).
        return fail(StatusCode::invalid_argument,
                    "native Tile FP8 GEMM requires a non-empty residual view (a 1-element dummy "
                    "when the epilogue is none)");
    }
    if (uint64_t(y.size()) < need_y) {
        return fail(StatusCode::invalid_argument, "native Tile FP8 GEMM output view is too small",
                    luisa::format("y.size {} < m*n {}", y.size(), need_y));
    }
    luisa::string shader_error;
    auto *shader = _shader_for(m, n, k, residual_epilogue, shader_error);
    if (shader == nullptr) {
        return fail(StatusCode::internal_error, "native Tile FP8 GEMM shader compile failed",
                    std::move(shader_error));
    }
    // Same bytes, same offset, element type only: the packed uint32 FP8 buffer is
    // reinterpreted in place. There is no copy and no repack anywhere.
    auto weights = w.as<ct::float8_e4m3fn>();
    if (uint64_t(weights.size()) < uint64_t(n) * k) {
        return fail(StatusCode::invalid_argument,
                    "native Tile FP8 GEMM FP8 view is too small after reinterpretation",
                    luisa::format("{} < n*k {}", weights.size(), uint64_t(n) * k));
    }
    return (*shader)(x, weights, scales, residual, y).dispatch();
}

}// namespace luisa::qwen::dsl
