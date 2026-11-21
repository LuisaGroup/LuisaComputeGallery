#pragma once
// Qwen3.5-4B tensor-core PREFILL GEMM through the CUDA native-shader route
// (opt_plan/09 N1a). Namespace luisa::qwen::dsl.
//
// This is the CUDA-only accelerator of the prefill projection GEMM: a real
// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` kernel, hand-written in
// CUDA C++, compiled at run time by NVRTC through `NativeShaderExt`
// (`NativeShaderLanguage::CUDA_NVRTC`) and dispatched as a
// `NativeShaderDispatchCommand`. It is NOT part of the DSL/AST-to-XIR path, it
// does not exist on dx/vk (those keep `make_qwen35_gemm_mt_kernel`), and it is
// not tile::mma (the separate, strictly opt-in CUDA 13.5 native-Tile FP8
// experiment lives in cuda_tile_gemm.h).
//
// Scope, deliberately:
//  * GEMM ONLY (prefill rows >= 16). Decode keeps its GEMV specializations and
//    the static CUDA decode graph is untouched (native commands are never
//    recorded into it).
//  * The argument ABI is EXACTLY the DSL GEMM's: x [M,K] FP32, W [N,K] packed
//    BF16 (2 codes per uint32) or FP8 E4M3 (4 codes per uint32 + group scales),
//    residual [M,N] FP32 (dummy when the epilogue is `none`), y [M,N] FP32. No
//    weight repacking and no new buffers, so the same `QwenDeviceTensor` feeds
//    both paths and the two are directly differential-testable.
//  * Fail closed. `create()` returns a Status; anything the kernel cannot do
//    exactly (unsupported format, K that is not even, a geometry the tiling
//    cannot index, a missing extension, an NVRTC/toolchain failure) disables the
//    whole object and the engine keeps the DSL MT path. There is no per-shape
//    fallback in the middle of a prompt.
//
// Numerics: the device code rounds the ACTIVATION FP32->BF16 once, at the
// shared-tile boundary (see native_gemm_kernel.h); weights are consumed exactly
// as stored and the accumulation is FP32. Against the FP32-activation reference
// (`ref_qwen_matmul`, and the DSL MT kernel) that is a ~1.6e-3 relative RMS /
// ~2e-3 relative max deviation in the measured shapes - the documented tolerance
// in docs/examples/example_qwen.md - not a bit-exact path.
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>

#include "qwen/common.h"
#include "qwen/dsl_ops.h"

namespace luisa::qwen::dsl {

// Block tiling of the tensor-core kernel. The warp tile is fixed at 32x32 output
// elements (2 `m16` x 4 `n8` mma fragments, 32 FP32 accumulators per thread);
// the block tile is `block_m x block_n`, `warps` 32-lane warps cover it, and
// `block_k` is the shared-memory k step of the double-buffered pipeline.
// `block_m=128, block_n=64, block_k=32, warps=8` is the measured default: at
// M=1024,N=9216,K=2560 it reaches ~74 TFLOPS (52 TFLOPS for the 64x64/4-warp
// tile) because it halves the number of weight passes over DRAM/L2.
struct QwenNativeGemmConfig {
    uint32_t block_m{128u};
    uint32_t block_n{64u};
    uint32_t block_k{32u};
    uint32_t warps{8u};
    // FP8 E4M3 quantization group of the packed scales; BAKED INTO the generated
    // source (the scale index becomes a constant shift), so a kernel compiled for
    // one group cannot be handed another group's scale buffer. Ignored by BF16.
    uint32_t fp8_group{64u};
    [[nodiscard]] constexpr uint32_t block() const noexcept { return warps * 32u; }
};

// Smallest prefill row count the tensor-core kernel takes. Below it one mma
// fragment row set is not filled and the decode/MTP GEMV family already covers
// the shape, so the engine keeps the DSL path.
inline constexpr uint32_t qwen_native_gemm_min_rows = 16u;

// Narrowest N at which the tensor-core kernel beats the DSL MT kernel. MEASURED,
// not assumed (benchmark_qwen_decode --prefill-gemm, bf16, M = 64/256/1024):
//  * N = 32   (the GDN gate/alpha/beta pair): with BF16 weights the DSL MT kernel
//    wins at every M (M=1024: mt 0.0499 ms vs native 0.0615 ms) - a 64-wide block
//    tile is half empty, so the launch is fixed-cost bound and the weight tile is
//    reused only 2x. With FP8 the two are within 5 % either way (0.0725 vs
//    0.0692 ms), which is inside the decision noise and keeps one rule for both
//    formats;
//  * N >= 1024: the tensor-core kernel wins by 1.8x (M=64) to 10.7x (M=1024) on
//    N=1024 and by 10-16x on N = 2560/8192/9216.
// 256 is the conservative point between the two: it is already past the measured
// loss and leaves a margin for shapes that were not measured.
inline constexpr uint32_t qwen_native_gemm_min_columns = 256u;

[[nodiscard]] constexpr bool qwen_native_gemm_prefers_columns(uint64_t n) noexcept {
    return n >= qwen_native_gemm_min_columns;
}

// Geometry of one launch. `dispatch` is 2D exactly like the DSL GEMM's:
// grid.x tiles M (block_m rows), grid.y tiles N (block_n columns). The native
// dispatch command takes a THREAD count, so threads_x = blocks_x * block and the
// launcher's block size derives grid.x again.
struct QwenNativeGemmPlan {
    uint32_t blocks_x{0u};
    uint32_t blocks_y{0u};
    uint32_t block{0u};
    uint32_t threads_x{0u};
    uint64_t blocks{0u};
    [[nodiscard]] bool empty() const noexcept { return blocks == 0u; }
};

// The two formats the packed ABI can hand to a tensor-core kernel. BF16 is
// native (the stored codes ARE the mma operands). FP8 E4M3 is widened to BF16 -
// and scaled - on the way into the shared tile, so the weight buffers and the
// scale groups are untouched and no repacking happens; that variant is slower to
// stage (it cannot use cp.async for the weights) and its numbers are reported
// separately in docs/examples/example_qwen.md.
[[nodiscard]] bool qwen_native_gemm_supported_format(QwenWeightFormat format) noexcept;

[[nodiscard]] Status validate_qwen_native_gemm_config(const QwenNativeGemmConfig &cfg) noexcept;

// Fail-closed geometry gate for one projection: M (rows), N (outputs), K
// (reduction) and the format. K must be EVEN (the packed-word ABI) - the device
// code then picks a 16-byte cp.async per 8 codes when K % 8 == 0 and a 4-byte
// one otherwise, and zero-fills the k tail.
[[nodiscard]] Status validate_qwen_native_gemm_shape(uint64_t m, uint64_t n, uint64_t k,
                                                     QwenWeightFormat format,
                                                     uint32_t fp8_group = 64u) noexcept;

[[nodiscard]] Result<QwenNativeGemmPlan> plan_qwen_native_gemm_dispatch(
    uint64_t m, uint64_t n, uint64_t k, const QwenNativeGemmConfig &cfg) noexcept;

// The CUDA C++ translation unit this module hands to NVRTC. Exposed so the host
// tests can assert the generated text carries the requested tiling and so a
// failure can be printed.
// `format` selects the weight realization baked into the source (QN_FP8).
[[nodiscard]] luisa::string qwen_native_gemm_cuda_source(const QwenNativeGemmConfig &cfg,
                                                        bool residual,
                                                        QwenWeightFormat format = QwenWeightFormat::bf16);

// Owner of the compiled tensor-core GEMM variants of one weight format.
//
// Lifetime: `create()` compiles through `NativeShaderExt`, `reset()`/
// ~QwenNativeGemm destroy the `NativeShader` instances (strictly before the
// `Device`), and every dispatch must have completed before that - the engine
// synchronizes before it tears the engine down, exactly like its DSL shaders.
// Non-copyable, non-movable: it is held by value by the engine.
class QwenNativeGemm {
private:
    compute::NativeShaderExt *_ext{nullptr};
    compute::NativeShader _plain;
    compute::NativeShader _residual;
    QwenNativeGemmConfig _cfg{};
    QwenWeightFormat _format{QwenWeightFormat::bf16};
    compute::NativeShaderResourceBinding _bindings[5]{};
    uint32_t _binding_count{0u};
    bool _fp8{false};
    double _jit_ms{0.0};
    luisa::string _entry_point;

public:
    QwenNativeGemm() noexcept = default;
    ~QwenNativeGemm() noexcept { reset(); }
    QwenNativeGemm(const QwenNativeGemm &) = delete;
    QwenNativeGemm &operator=(const QwenNativeGemm &) = delete;

    // Compile the plain and the residual variant of `format`. Fails closed: on
    // any failure the object stays invalid (`valid() == false`) and `detail`
    // carries the NVRTC/reflection message. `jit_ms` records the measured COLD
    // compile cost of this call (both variants), which is what a caller reports
    // honestly as start-up latency.
    [[nodiscard]] Status create(compute::Device &device, QwenWeightFormat format,
                                const QwenNativeGemmConfig &cfg) noexcept;
    void reset() noexcept;

    [[nodiscard]] bool valid() const noexcept { return _binding_count != 0u; }
    [[nodiscard]] bool has_residual() const noexcept { return _residual.operator bool(); }
    [[nodiscard]] bool is_fp8() const noexcept { return _fp8; }
    [[nodiscard]] const QwenNativeGemmConfig &config() const noexcept { return _cfg; }
    [[nodiscard]] QwenWeightFormat format() const noexcept { return _format; }
    [[nodiscard]] double jit_ms() const noexcept { return _jit_ms; }
    [[nodiscard]] luisa::span<const compute::NativeShaderResourceBinding> bindings() const noexcept {
        return luisa::span<const compute::NativeShaderResourceBinding>{_bindings, _binding_count};
    }

    // Build one dispatch command. The views must cover the whole GEMM:
    // `x` [m,k] FP32, `w` [n,k] packed, `scales` (FP8 only, any size for BF16),
    // `residual` [m,n] FP32 when `residual_epilogue` else a dummy 1-element view,
    // `y` [m,n] FP32. Returns nullptr and fills `error` on a geometry the tiling
    // cannot index - callers validate with validate_qwen_native_gemm_shape() at
    // create time, so this is a defensive check, not a fallback path.
    [[nodiscard]] luisa::unique_ptr<compute::Command> build_dispatch(
        bool residual_epilogue,
        const compute::BufferView<float> &x,
        const compute::BufferView<uint32_t> &w,
        const compute::BufferView<float> &scales,
        const compute::BufferView<float> &residual,
        const compute::BufferView<float> &y,
        uint32_t m, uint32_t n, uint32_t k,
        luisa::string *error = nullptr) const noexcept;
};

}// namespace luisa::qwen::dsl
