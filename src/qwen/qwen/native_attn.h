#pragma once
// Qwen3.5-4B causal GQA PREFILL ATTENTION through the CUDA native-shader route
// (opt_plan/09 N1b). Namespace luisa::qwen::dsl.
//
// This is the CUDA-only accelerator of the prefill attention: a real
// FlashAttention-2 kernel (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
// for both QK^T and P.V, FP32 online softmax, FP32 accumulators), hand-written in
// CUDA C++, compiled at run time by NVRTC through `NativeShaderExt`
// (`NativeShaderLanguage::CUDA_NVRTC`) and dispatched as a
// `NativeShaderDispatchCommand`. It is NOT part of the DSL/AST-to-XIR path, it does
// not exist on dx/vk (those keep `make_qwen35_attn_prefill_tiled_kernel`), and it is
// not tile::mma.
//
// Scope, deliberately:
//  * PREFILL ONLY (rows >= 1). Decode keeps its block-cooperative kernel, the
//    split-KV family and the static CUDA decode graph untouched (native commands
//    are never recorded into it).
//  * The argument ABI is EXACTLY the DSL tiled kernel's: q/gate [tokens, q_heads,
//    head_dim] FP32, k_cache/v_cache [capacity, kv_heads, head_dim] FP32, out
//    [tokens, q_heads, head_dim] FP32, plus (tokens, scale, pos0, kv_len, capacity)
//    as runtime scalars. No repacking, no layout change, no new buffer: the same
//    `QwenDeviceTensor`/KV cache feeds both paths, which is what makes them
//    directly differential-testable.
//  * The model geometry (head_dim 256, q_heads 16, kv_heads 4, i.e. a 4:1 GQA pack)
//    is a COMPILE-TIME constant of the generated kernel, exactly like the DSL tiled
//    kernel. A model with another geometry disables the path instead of falling back
//    per dispatch.
//  * Fail closed. `create()` returns a Status; anything the kernel cannot do
//    exactly (a different geometry, a device without the extension, an NVRTC or
//    reflection failure, a shared-memory budget below the tile) disables the whole
//    object and the engine keeps the DSL tiled kernel. There is no per-position or
//    per-length JIT: one compiled kernel serves every prompt length, position,
//    ragged tail and cache capacity, because tokens/pos0/kv_len/capacity stay
//    runtime scalars.
//
// Numerics (see native_attn_kernel.h for the device-side detail): Q and K are
// rounded FP32 -> BF16 once, the P = exp(...) weights are rounded to BF16 for the
// P.V mma, and both accumulations are FP32 with an FP32 running max/sum. Against
// the FP32 DSL tiled kernel and `ref_qwen_attn_prefill` this is a documented
// tolerance (relative RMS ~1e-3, relative max ~1e-2 on well-conditioned data), NOT
// bit equality; docs/examples/example_qwen.md carries the measured numbers.
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

// Block tiling of the tensor-core attention kernel.
//
// `head_dim = 256` is what shapes every other choice: one m16n8k16 fragment needs
// 16 query rows, and a [16 x 256] FP32 accumulator is 128 registers per lane, so a
// warp can hold EITHER its Q fragments OR a full-width output accumulator. The
// kernel therefore splits the head dimension across a WARP PAIR (each warp owns
// 128 channels, i.e. 64 accumulator registers per lane) and exchanges the two
// halves of the QK^T partial through one shared [q_tile x kv_tile] FP32 buffer per
// q head. Both warps of the pair then run the identical FP32 softmax and build the
// identical BF16 P fragments, so the result is deterministic.
//
//  * q_tile: 16 query tokens per block (one m16 fragment). Fixed: two m16 fragments
//    would double the accumulator to 128 registers per lane.
//  * kv_tile: 32 keys per iteration. Fixed by the 48 KiB per-block STATIC shared
//    memory limit of the native-shader launch path (it requests no dynamic shared
//    memory): 2 x 32 x 264 x 2 B of BF16 K/V tiles + 4 x 16 x 33 x 4 B of score
//    exchange = 42.25 KiB. kv_tile = 64 would need 66 KiB for the K/V tiles alone.
//  * The 4 q heads of one kv head are packed into one block (2 warps each), so one
//    K/V staging serves 4 x 16 query rows - the GQA dedup the DSL tiled kernel gets
//    from its 4 warps.
struct QwenNativeAttnConfig {
    uint32_t head_dim{256};
    uint32_t q_heads{16};
    uint32_t kv_heads{4};
    uint32_t q_tile{16};
    uint32_t kv_tile{32};
    // q heads per kv head (the GQA pack of one block) and the warp count: one
      // warp pair per q head because of the head-dim split above.
    [[nodiscard]] constexpr uint32_t gqa() const noexcept {
        return kv_heads == 0u ? 0u : q_heads / kv_heads;
    }
    [[nodiscard]] constexpr uint32_t warps() const noexcept { return gqa() * 2u; }
    [[nodiscard]] constexpr uint32_t block() const noexcept { return warps() * 32u; }
};

// Shared-memory bytes one block of this configuration declares STATICALLY. The
// native launch path passes 0 dynamic bytes, so this must stay <= 48 KiB (the
// sm_80+ per-block static limit) and the host gate below enforces it.
[[nodiscard]] uint32_t qwen_native_attn_shared_bytes(const QwenNativeAttnConfig &cfg) noexcept;

// Query rows below which the DSL tiled kernel keeps prefill attention. MEASURED with
// benchmark_qwen_decode --prefill-attention (median of 30, one layer, full cache):
// the native kernel is SLOWER than the DSL tiled kernel at S = 1, 8 and 16
// (0.0167/0.0173/0.0216 ms vs 0.0116/0.0152/0.0185 ms: a 4-block launch of
// 16-token tiles with a single 32-key tile per block is all fixed cost and 15/16 of
// each query tile is masked off) and FASTER from S = 32 on (0.0219 vs 0.0256 ms, then
// 1.5x at 64, 2.5x at 128, 7.9x at 1024 and 10.1x at 4096). 32 is that measured
// crossover; below it the engine keeps the DSL tiled kernel, and a caller that wants
// the native kernel for a smaller tile can dispatch it directly (the device test does).
inline constexpr uint32_t qwen_native_attn_min_rows = 32u;

[[nodiscard]] constexpr bool qwen_native_attn_prefers_rows(uint64_t rows) noexcept {
    return rows >= qwen_native_attn_min_rows;
}

[[nodiscard]] Status validate_qwen_native_attn_config(const QwenNativeAttnConfig &cfg) noexcept;

// Fail-closed input gate: the model geometry (head_dim/q_heads/kv_heads), the
// static shared-memory ceiling, the 32-bit scalar ABI and the same causal/capacity
// contract the DSL kernels accept (`validate_qwen_attn_prefill_inputs`).
[[nodiscard]] Status validate_qwen_native_attn_inputs(uint64_t tokens, uint64_t pos0, uint64_t kv_len,
                                                      uint64_t capacity,
                                                      const QwenNativeAttnConfig &cfg) noexcept;

// Geometry of one launch: grid.x tiles the tokens (`q_tile` rows per block),
// grid.y is the kv head. The native dispatch command takes a THREAD count, so
// threads_x = blocks_x * block and the launcher's block size derives grid.x again.
struct QwenNativeAttnPlan {
    uint32_t blocks_x{0u};
    uint32_t blocks_y{0u};
    uint32_t block{0u};
    uint32_t threads_x{0u};
    uint64_t blocks{0u};
    [[nodiscard]] bool empty() const noexcept { return blocks == 0u; }
};

[[nodiscard]] Result<QwenNativeAttnPlan> plan_qwen_native_attn_dispatch(
    uint64_t tokens, const QwenNativeAttnConfig &cfg) noexcept;

// The CUDA C++ translation unit this module hands to NVRTC. Exposed so the host
// tests can assert the generated text carries the requested tiling and so a
// failure can be printed.
[[nodiscard]] luisa::string qwen_native_attn_cuda_source(const QwenNativeAttnConfig &cfg);

// Owner of the compiled tensor-core attention kernel of one engine.
//
// Lifetime: `create()` compiles through `NativeShaderExt`, `reset()`/
// ~QwenNativeAttn destroy the `NativeShader` instance (strictly before the
// `Device`), and every dispatch must have completed before that - the engine
// synchronizes before it tears the engine down, exactly like its DSL shaders.
// Non-copyable, non-movable: it is held by value by the engine.
class QwenNativeAttn {
private:
    compute::NativeShaderExt *_ext{nullptr};
    compute::NativeShader _shader;
    QwenNativeAttnConfig _cfg{};
    compute::NativeShaderResourceBinding _bindings[5]{};
    uint32_t _binding_count{0u};
    double _jit_ms{0.0};
    luisa::string _entry_point;

public:
    QwenNativeAttn() noexcept = default;
    ~QwenNativeAttn() noexcept { reset(); }
    QwenNativeAttn(const QwenNativeAttn &) = delete;
    QwenNativeAttn &operator=(const QwenNativeAttn &) = delete;

    // Compile the single kernel variant. Fails closed: on any failure the object
    // stays invalid (`valid() == false`) and `detail` carries the NVRTC/reflection
    // message. `jit_ms` records the measured COLD compile cost, which is what a
    // caller reports honestly as start-up latency (native shaders bypass
    // ShaderSerializer, so there is no on-disk cache).
    [[nodiscard]] Status create(compute::Device &device, const QwenNativeAttnConfig &cfg) noexcept;
    void reset() noexcept;

    [[nodiscard]] bool valid() const noexcept { return _binding_count != 0u; }
    [[nodiscard]] const QwenNativeAttnConfig &config() const noexcept { return _cfg; }
    [[nodiscard]] double jit_ms() const noexcept { return _jit_ms; }
    [[nodiscard]] uint32_t shared_bytes() const noexcept { return qwen_native_attn_shared_bytes(_cfg); }
    [[nodiscard]] luisa::span<const compute::NativeShaderResourceBinding> bindings() const noexcept {
        return luisa::span<const compute::NativeShaderResourceBinding>{_bindings, _binding_count};
    }

    // Build one dispatch command. The views must cover the whole call: q/gate/out
    // [tokens, q_heads, head_dim], k_cache/v_cache [capacity, kv_heads, head_dim].
    // Returns nullptr and fills `error` on a geometry the tiling cannot index -
    // callers validate with validate_qwen_native_attn_inputs() before dispatching, so
    // this is a defensive check, not a fallback path.
    [[nodiscard]] luisa::unique_ptr<compute::Command> build_dispatch(
        const compute::BufferView<float> &q,
        const compute::BufferView<float> &gate,
        const compute::BufferView<float> &k_cache,
        const compute::BufferView<float> &v_cache,
        const compute::BufferView<float> &out,
        uint32_t tokens, float scale, uint32_t pos0, uint32_t kv_len, uint32_t capacity,
        luisa::string *error = nullptr) const noexcept;
};

}// namespace luisa::qwen::dsl
