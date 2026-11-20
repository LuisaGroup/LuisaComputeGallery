#pragma once
// Public text-only, single-sequence Qwen3.5-4B inference engine interface.
// A prefill consumes every prompt token and returns logits for the next token.
// decode consumes one previously selected token and returns the following
// logits. Implementations own request state; weights are borrowed and immutable.
// Engines and their staging/session state are single-threaded objects: calls on
// one engine and its associated GGUF upload/stream must be externally serialized.
// Public execution/reset calls must use the same Stream supplied to create();
// sparse KV residency and ordered teardown intentionally rely on that identity.

#include <cstdint>
#include <memory>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "qwen/benchmark.h"
#include "qwen/caps.h"
#include "qwen/weights.h"

namespace luisa::qwen {

enum class QwenKernelKind : uint32_t {
    dsl = 0,
    tile = 1,
};
[[nodiscard]] luisa::string_view qwen_kernel_kind_name(QwenKernelKind kind) noexcept;

struct QwenEngineOptions {
    QwenKernelKind kernel{QwenKernelKind::dsl};
    // Logical full-attention KV token capacity, reserved virtually and backed
    // by sparse physical tiles on demand. Must be >0 and <= the model's native
    // context length. GDN state is fixed-size and allocated separately.
    uint32_t max_context{4096};
    // KV sparse ALLOCATION unit in bytes (0 = the allocator's 2 MiB default).
    // One heap is allocated per page, so this trades growth granularity for
    // allocation count: DX/VK cost one D3D12MA/VkDeviceMemory allocation per
    // page and Vulkan's maxMemoryAllocationCount is commonly 4096, so raise this
    // for very long contexts on Vulkan (the allocator warns with the exact value
    // it would need). Must be a multiple of the probed sparse tile size; a
    // non-conforming value fails closed with invalid_argument.
    uint64_t kv_page_bytes{0};
    // Number of prompt rows per engine prefill call. DSL subdivides these rows
    // into serial 64-token WY/UT chunks; this is an activation/launch tradeoff.
    uint32_t prefill_chunk{8};
    // DSL prefill uses the FP32 WY/UT chunk-64 GDN algorithm by default.
    // Set false for the exact per-token recurrence as a differential/debug
    // path. Decode always uses the recurrence. Tile currently has its own
    // recurrent GDN schedule and does not consult this DSL-specific switch.
    bool dsl_gdn_wy_ut{true};
    // Single-row decode specializations change reduction order. MTP forces this
    // false so verify, accepted-prefix replay, and plain reference decode share
    // the original numerical path. Ordinary optimized decode keeps it true.
    bool dsl_decode_specialized{true};
    // FP8 CUDA bounded device-greedy decode replays a whole-step graph when
    // CudaGraphExt is available. Capture/update failures are recoverable and
    // never execute a partially consumed list. Other variants are unchanged.
    bool dsl_cuda_graph{true};
    // Opt-in command-level profiling. False keeps ordinary batched command
    // lists. True uses the supplied stream to commit + synchronize after every
    // kernel/transfer and exposes raw host-wall samples via benchmark_report().
    bool benchmark{false};
};

struct QwenEngineInfo {
    QwenKernelKind kernel{QwenKernelKind::dsl};
    WeightStorage weight_storage{WeightStorage::bf16_packed};
    uint32_t position{0};
    uint32_t max_context{0};
    uint32_t prefill_chunk{0};
    // Configured logical KV payload + fixed GDN state bytes. Sparse alignment
    // padding and time-varying physical residency are allocator properties.
    uint64_t cache_bytes{0};
    uint64_t workspace_bytes{0};
    uint64_t compiled_shader_count{0};
    uint64_t cuda_graph_steps{0};
    uint64_t cuda_graph_updates{0};
    bool gdn_wy_ut{false};
    // Device capability record captured at create (backend, warp size, sparse
    // tile, effective KV page geometry, graph/PDL/VK extension negotiation).
    // Carried here so a benchmark log is self-describing: "the accelerator did
    // not help" stays distinguishable from "the accelerator never engaged".
    QwenBackendCaps caps{};
};

class QwenEngine;

// All engine factories allocate through Luisa's allocator. The deleter must
// use the same allocator; EASTL's default_delete otherwise mixes CRT and
// mimalloc heaps in the gallery's static-library configuration.
struct QwenEngineDeleter {
    void operator()(QwenEngine *p) const noexcept;
};
using QwenEnginePtr = luisa::unique_ptr<QwenEngine, QwenEngineDeleter>;

class QwenEngine {
public:
    virtual ~QwenEngine() noexcept = default;
    QwenEngine(const QwenEngine &) = delete;
    QwenEngine &operator=(const QwenEngine &) = delete;

    // Create exactly the requested implementation. No DSL<->Tile fallback and
    // no BF16<->FP8 substitution. `weights` must outlive the returned engine.
    [[nodiscard]] static Result<QwenEnginePtr> create(
        compute::Device &device, compute::Stream &stream,
        const Qwen35Model &model, const QwenDeviceWeights &weights,
        const QwenEngineOptions &options);

    // Clear KV/GDN/conv state and position. The stream is synchronized before
    // returning so subsequent host actions observe a fully reset session.
    [[nodiscard]] virtual Status reset(compute::Stream &stream) = 0;

    // Consume a nonempty prompt and download FP32 logits for the next token.
    // The implementation synchronizes before returning. Input ids are checked
    // against the vocabulary and total context capacity.
    [[nodiscard]] virtual Status prefill(compute::Stream &stream,
                                         luisa::span<const uint32_t> prompt_tokens,
                                         luisa::vector<float> &logits_out) = 0;

    // Consume one selected token after a nonempty prefill/decode sequence and
    // download FP32 logits for the next token. Synchronizes before returning.
    [[nodiscard]] virtual Status decode(compute::Stream &stream, uint32_t input_token,
                                        luisa::vector<float> &logits_out) = 0;

    // Optional greedy-only in-flight decode. The first call passes a host input
    // token; later calls pass nullptr and consume the previous device argmax.
    // Each call records one forward without a host sync and writes the selected
    // token to history_slot. download_device_greedy performs the bounded sync
    // and validates the returned range. Unsupported engines keep the ordinary
    // synchronous logits API; temperature sampling never uses this path.
    [[nodiscard]] virtual bool supports_device_greedy_decode() const noexcept { return false; }
    [[nodiscard]] virtual Status decode_device_greedy(compute::Stream &, const uint32_t *,
                                                      uint32_t) {
        return Status::error(StatusCode::unsupported, "device greedy decode is not supported");
    }
    [[nodiscard]] virtual Status download_device_greedy(compute::Stream &, uint32_t,
                                                        luisa::span<uint32_t>) {
        return Status::error(StatusCode::unsupported, "device greedy download is not supported");
    }

    [[nodiscard]] virtual QwenEngineInfo info() const noexcept = 0;
    // Null when options.benchmark was false. Otherwise the latest
    // prefill/decode report, owned by the engine and valid until the next call.
    [[nodiscard]] virtual const QwenBenchmarkReport *benchmark_report() const noexcept {
        return nullptr;
    }

protected:
    QwenEngine() noexcept = default;
};

// Host greedy selection with deterministic lowest-index tie breaking. Kept
// outside the operators so all four variants share exactly the same sampler.
[[nodiscard]] Result<uint32_t> qwen_argmax(luisa::span<const float> logits) noexcept;

// Reproducible categorical sampling over FP32 logits. temperature == 0 is the
// exact qwen_argmax path; positive finite temperatures apply softmax(logits/T).
// `draw_index` is the absolute generated-token index, not a decode-call count:
// counter-based SplitMix64 randomness therefore stays reproducible across
// prefill chunking and speculative replay. No mutable RNG state is required.
struct QwenSampleOptions {
    float temperature{0.0f};
    uint64_t seed{0x51f15eED1234abcdull};
};
[[nodiscard]] Result<uint32_t> qwen_sample(
    luisa::span<const float> logits, const QwenSampleOptions &options,
    uint64_t draw_index) noexcept;

}// namespace luisa::qwen
