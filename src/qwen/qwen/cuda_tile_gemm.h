#pragma once
// Qwen3.5-4B PREFILL GEMM on the CUDA native Tile IR route, typed FP8 weight
// storage. Namespace luisa::qwen::dsl.
//
// WHAT THIS IS
// ------------
// ONE optional accelerator for the prefill projection GEMM, built from the
// Luisa Tile DSL (`tile::tile_kernel` -> `tile::Kernel` -> `tile::compile`,
// `Lowering::NATIVE`), i.e. the CUDA 13.4+ path that emits CUDA Tile C++, runs
// it through NVRTC's Tile IR backend and assembles it with `tileiras`. It is
// NOT the DSL MT kernel, NOT `tile::bridge::xir` (that route is DX/VK and
// explicitly rejects FP8 storage), NOT the hand-written NVRTC CUDA kernel of
// `native_gemm.h`. QwenDslEngine selects it only when the caller explicitly sets
// QwenEngineOptions::cuda_tile_fp8_prefill; it is never the default.
//
// WHAT IT CONSUMES (the EXISTING packed ABI, unchanged)
// ----------------------------------------------------
//  * x        [m, k] FP32  - activations, FP32 in DRAM exactly like the DSL GEMM
//  * w        [n, k] FP8 E4M3FN, four codes per little-endian uint32 word,
//             low element first - the SAME `QwenDeviceTensor` FP8 weight buffer
//             the current native/DSL GEMMs read. The kernel takes the packed
//             `BufferView<uint32_t>` and reinterprets it with
//             `.as<tile::float8_e4m3fn>()`, so the allocation, its offset and
//             every byte are IDENTICAL to what the other paths see: there is no
//             host repack, no widened copy and no dequantized buffer.
//  * scales   [n, k/64] FP32 - one FP32 scale per contiguous group of 64 K
//             elements per output row (group fixed at 64).
//  * residual [m, n] FP32 (a 1-element dummy when the epilogue is `none`)
//  * y        [m, n] FP32
//
// KERNEL AND NUMERICS (deliberate, documented, not a bit-exact path)
// -----------------------------------------------------------------
// For every (m, n) output tile and every K block of 64:
//   1. load the FP32 activation tile and the TYPED FP8 weight tile (a real
//      `TensorView<const tile::float8_e4m3fn, 2>` load, not a packed-word
//      software decode);
//   2. cast BOTH tiles to `tile::bfloat16`. The weight cast is EXACT: E4M3 has
//      3 mantissa bits and BF16 has 7, and every finite E4M3 magnitude is
//      representable; the activation cast is the ONE lossy rounding of the
//      kernel (FP32 -> BF16, round-to-nearest-even), exactly like the measured
//      native CUDA tensor-core kernel;
//   3. `partial = mma(a_bf16, w_bf16, zeros<f32>, MmaPolicy{allow_reassociation
//      = true})`, i.e. one BF16 tensor-core partial dot per 64-element K block
//      with an FP32 accumulator;
//   4. `acc += partial * scale[n, block]` - the FP32 group scale is applied to
//      the BF16 PARTIAL DOT in FP32, AFTER the contraction, never to the
//      individual weights. That is why no narrow arithmetic appears anywhere:
//      the scale multiplies an FP32 tile.
// After the K loop the optional residual is added ONCE and the tile is stored.
//
// Against the FP32-activation reference (`ref_qwen_matmul`) the deviation is the
// single BF16 rounding of the activation plus FP32 reassociation; against an
// oracle that rounds the activation the same way (BF16 activations, BF16(fp8)
// weights, double-precision partials scaled by the FP32 group scale) the kernel
// agrees to FP32 accumulation order. `tests/integration/qwen/test_qwen_native_gemm.cpp`
// measures both and reports the numbers; this header never claims bit-exactness.
//
// FAIL CLOSED
// -----------
//  * FP8 E4M3FN storage ONLY. BF16/Q8_0/W4 weights are refused (`unsupported`):
//    this module never re-encodes a weight buffer into another format.
//  * K must be a positive multiple of 64 (tile_k == scale group == 64), so there
//    is no K tail and no partial scale group: ragged M/N are handled by the Tile
//    `bounds::zero` load/store masking, a ragged K is refused. A ragged M is
//    always allowed (the activation and output tiles are FP32). A ragged N needs
//    a PARTIAL typed-FP8 weight tile load, which the toolkit only compiles from
//    sm_90 up: the create-time probe measures that, and `build_dispatch` refuses
//    a ragged N (with the reason) where it is unavailable instead of producing a
//    shader the driver would reject.
//  * m, n, k must be positive; the extents, the per-buffer element counts and
//    the root grid must be statically representable (see the constants below).
//  * `create()` requires the `cuda` backend AND the run-time opt-in
//    `LUISA_CUDA_TILE_IR=1`. Missing/a wrong value is an explicit error, never a
//    silent TIRX or DSL fallback. A tiny typed-FP8 compile probe runs at create
//    time so a CUDA < 13.4 toolkit, a missing `tileiras`, or a target the native
//    Tile IR backend cannot serve fails at start-up instead of mid-prompt.
//  * Every compile is lazy per (m, n, k, epilogue) static shape and returns a
//    diagnostic instead of a shader; `build_dispatch` then returns nullptr with
//    the same text. Buffer sizes are checked on the host BEFORE the dispatch is
//    built, so bad user input never reaches a runtime assertion.
//
// COST
// ----
// There is no shape-independent shader: the Tile IR needs the extents, so the
// FIRST dispatch of every (m, n, k, epilogue) compiles (NVRTC Tile IR + tileiras).
// Measured on the gallery's machine: ~1.0 s per static shape (12 shapes -> roughly 10-15 s),
// which is a real start-up/TTFT cost - compile the shapes a prompt will need before
// the request path and report `compile_count()`/`jit_ms()` rather than hiding it.
// The create-time probes add ~0.35 s once.
//
// LIFETIME
// --------
// `create()` / the lazy compiles borrow the `Device`; `reset()`/~QwenCudaTileGemm
// destroy the cached `tile::Shader` instances. Both must happen while the Device
// is alive and after the last dispatch that used them completed - the same
// contract `QwenNativeGemm` documents.
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/tile.h>
#include <luisa/tile/runtime.h>

#include "qwen/common.h"
#include "qwen/dsl_ops.h"

namespace luisa::qwen::dsl {

namespace ct = ::luisa::compute::tile;

// The K step IS the FP8 quantization group. Both are fixed: the kernel keeps one
// scale per K block, so a different group would need a different kernel shape
// and a different weight ABI. Fail closed instead of silently re-grouping.
inline constexpr uint32_t qwen_cuda_tile_gemm_k_step = 64u;
inline constexpr uint32_t qwen_cuda_tile_gemm_group = 64u;

// Static bounds. They are deliberately conservative and independently checked:
//  * every extent <= 2^20 (the model's largest is N = 9216);
//  * every per-buffer ELEMENT count <= 2^32 - 1 and every byte count < 2^63;
//  * the root program grid (ceil(m/tile_m) * ceil(n/tile_n)) <= 2^31 - 1.
inline constexpr uint64_t qwen_cuda_tile_gemm_max_extent = 1ull << 20u;
inline constexpr uint64_t qwen_cuda_tile_gemm_max_elements = 0xffffffffull;
// Tile-shape limits of the native CUDA Tile codegen: power-of-two extents and a
// per-tile element volume <= 2^24. The config validator keeps a much smaller
// budget (see kMaxTileElements) so a typo cannot request an absurd shader.
inline constexpr uint32_t qwen_cuda_tile_gemm_min_tile = 16u;
inline constexpr uint32_t qwen_cuda_tile_gemm_max_tile = 128u;
inline constexpr uint32_t qwen_cuda_tile_gemm_max_tile_elements = 16384u;

// Block tiling of the native Tile kernel. `tile_m`/`tile_n` are the per-program
// output tile (power of two by construction); `tile_k` and `group` are fixed at
// 64. 64x64x64 is the default: at M=1024/N=9216/K=2560 the FP8 weight tile is
// reused across 16 M-programs and the activation tile across 144 N-programs.
struct QwenCudaTileGemmConfig {
    uint32_t tile_m{64u};
    uint32_t tile_n{64u};
    uint32_t tile_k{qwen_cuda_tile_gemm_k_step};
    uint32_t group{qwen_cuda_tile_gemm_group};
};

// FP8 E4M3FN only. Everything else keeps the existing DSL MT / native-shader
// path: this module has no ordered/narrow substitute for those formats.
[[nodiscard]] bool qwen_cuda_tile_gemm_supported_format(QwenWeightFormat format) noexcept;

// Device-free config gate: power-of-two tile_m/tile_n in [16, 128], a tile
// volume within the native Tile budget, and tile_k == group == 64.
[[nodiscard]] Status validate_qwen_cuda_tile_gemm_config(const QwenCudaTileGemmConfig &cfg) noexcept;

// Device-free shape gate for one projection. `m`/`n` are the dispatch extents
// (both must be nonzero: a static Tile capture of a zero extent is not a shape),
// `k` must be a positive multiple of 64, and the format must be FP8 E4M3FN.
[[nodiscard]] Status validate_qwen_cuda_tile_gemm_shape(uint64_t m, uint64_t n, uint64_t k,
                                                        QwenWeightFormat format,
                                                        uint32_t group = qwen_cuda_tile_gemm_group) noexcept;

// True when the process opted into the CUDA native Tile IR route
// (`LUISA_CUDA_TILE_IR=1`). Exposed so callers/tests can decide to skip the
// accelerator WITHOUT touching the global environment; `create()` applies the
// same check and reports its own explicit error.
[[nodiscard]] bool qwen_cuda_tile_gemm_native_route_requested() noexcept;

// One captured, statically specialized Tile kernel plus its diagnostics.
// `capture.ok()` implies `kernel.has_value() && kernel->valid()`; the caller
// compiles it with `tile::compile(device, *kernel, ...)` and must check
// `shader.metadata().error` there (see QwenCudaTileGemm::build_dispatch).
struct QwenCudaTileGemmCapture {
    Status status;
    luisa::optional<ct::Kernel> kernel;
    luisa::vector<luisa::string> diagnostics;

    [[nodiscard]] bool ok() const noexcept { return status.ok() && kernel.has_value() && kernel->valid(); }
};

// Device-free capture of the plain or the residual variant for one static shape.
// This is the exact kernel QwenCudaTileGemm compiles and dispatches; it is public
// so the host tests can verify the specialization (names, diagnostics) without a
// GPU and so a compile failure can be printed verbatim.
[[nodiscard]] QwenCudaTileGemmCapture capture_qwen_cuda_tile_gemm(
    const QwenCudaTileGemmConfig &cfg, uint32_t m, uint32_t n, uint32_t k, bool residual_epilogue) noexcept;

// Owner of the compiled native-Tile FP8 GEMM variants of one tiling.
//
// The shaders are compiled LAZILY, one per (m, n, k, epilogue) static shape and
// cached: the Tile IR needs the extents at capture time, so there is no
// shape-independent shader to compile up front. Every compile is counted and
// timed (`compile_count`, `jit_ms`) because a compile on the request path is a
// real start-up/TTFT cost the engine must be able to report.
class QwenCudaTileGemm {
private:
    // Borrowed; used only for lazy compiles. Outlives this object by contract.
    compute::Device *_device{nullptr};
    QwenCudaTileGemmConfig _cfg{};
    // key = residual(1 bit) | m(21) | n(21) | k(21); only static shapes enter.
    luisa::unordered_map<uint64_t, luisa::unique_ptr<ct::Shader>> _shaders;
    bool _ragged_fp8_supported{false};
    uint32_t _compile_count{0u};
    uint32_t _cache_hits{0u};
    double _jit_ms{0.0};
    double _probe_ms{0.0};

    [[nodiscard]] static uint64_t _key(uint32_t m, uint32_t n, uint32_t k, bool residual) noexcept;
    [[nodiscard]] Status _probe(compute::Device &device) noexcept;
    [[nodiscard]] ct::Shader *_shader_for(uint32_t m, uint32_t n, uint32_t k, bool residual,
                                          luisa::string &error) noexcept;

public:
    QwenCudaTileGemm() noexcept = default;
    ~QwenCudaTileGemm() noexcept { reset(); }
    QwenCudaTileGemm(const QwenCudaTileGemm &) = delete;
    QwenCudaTileGemm &operator=(const QwenCudaTileGemm &) = delete;

    // Validate the config, require cuda + LUISA_CUDA_TILE_IR=1, then run a tiny
    // typed-FP8 Tile compile probe. Any failure leaves the object invalid
    // (`valid() == false`) with `detail` carrying the backend/NVRTC/tileiras text.
    [[nodiscard]] Status create(compute::Device &device,
                                const QwenCudaTileGemmConfig &cfg = {}) noexcept;
    void reset() noexcept;

    [[nodiscard]] bool valid() const noexcept { return _device != nullptr; }
    [[nodiscard]] const QwenCudaTileGemmConfig &config() const noexcept { return _cfg; }

    // Measured capability: can this device/toolkit compile a PARTIAL (masked)
    // typed-FP8 tile load? Every ragged N needs one; ragged M does not (its tiles
    // are FP32), and neither does any N that is a multiple of `config().tile_n`.
    // The create-time probes decide it (it is NOT guessed from an SM number), and
    // `build_dispatch` refuses a ragged N when this is false - the measured
    // answer on SM89 is false: "the __nv_fp8_e4m3 types are not supported in tile
    // code for architectures below sm_90" comes straight from NVRTC.
    [[nodiscard]] bool supports_ragged_columns() const noexcept { return _ragged_fp8_supported; }

    // Reporting counters. `compile_count` includes nothing but lazy shape
    // compiles (the create-time probe is counted by `probe`/`probe_ms`).
    [[nodiscard]] uint32_t compile_count() const noexcept { return _compile_count; }
    [[nodiscard]] uint32_t cache_hits() const noexcept { return _cache_hits; }
    [[nodiscard]] uint32_t shader_count() const noexcept { return static_cast<uint32_t>(_shaders.size()); }
    [[nodiscard]] double jit_ms() const noexcept { return _jit_ms; }
    [[nodiscard]] double probe_ms() const noexcept { return _probe_ms; }
    [[nodiscard]] bool has_shader(uint32_t m, uint32_t n, uint32_t k, bool residual) const noexcept {
        return _shaders.contains(_key(m, n, k, residual));
    }

    // Build one dispatch command for
    //   y[m,n] = (sum_k x[m,k] * dequant(w)[n,k]) [+ residual[m,n]]
    // where `w` is the EXISTING packed `n*k/4` uint32 FP8 buffer and `scales` is
    // its `n*(k/64)` FP32 group scales. All views are checked on the host first;
    // a short view, a bad geometry or a failed lazy compile returns nullptr and
    // fills `error` (the engine keeps its own path). No silent fallback.
    [[nodiscard]] luisa::unique_ptr<compute::Command> build_dispatch(
        bool residual_epilogue,
        const compute::BufferView<float> &x,
        const compute::BufferView<uint32_t> &w,
        const compute::BufferView<float> &scales,
        const compute::BufferView<float> &residual,
        const compute::BufferView<float> &y,
        uint32_t m, uint32_t n, uint32_t k,
        luisa::string *error = nullptr) noexcept;
};

}// namespace luisa::qwen::dsl
