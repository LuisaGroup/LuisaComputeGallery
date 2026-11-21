#pragma once
// Conditional CUDA graph WHILE decode chunk (opt_plan/09 N4.2, opt_plan/03
// section 2.5): request validation + the DEDICATED instantiate/execute probe.
//
// Two independent things live here, and the engine needs BOTH before it will run
// a conditional WHILE:
//  1. qwen_validate_while_request() -- a host-only, device-free check of the
//     request itself (chunk size against the context, an EOS id that fits the
//     vocabulary, the token budget the body will consume). It is deliberately
//     free of device state so it can be unit-tested without a GPU.
//  2. qwen_probe_cuda_graph_while() -- a real instantiate + execute + replay
//     probe of a WHILE node built from scratch, measured on the running driver.
//     `CudaGraphExt::capabilities().conditional_nodes` is a TOOLKIT gate only (it
//     reports CUDA >= 12.3 and says nothing about the device), so a capability
//     flag is never sufficient evidence: the engine only enables the chunked
//     path when this probe has actually run a body the expected number of times,
//     replayed it, and honoured a zero-trip launch. Everything the probe cannot
//     verify is reported as a refusal with the measured reason, never assumed.
//
// Both are CUDA-only in effect: on any other backend the probe reports
// "CudaGraphExt not available" and the engine keeps the per-token graph.

#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "qwen/common.h"

namespace luisa::qwen {

// Host-side description of one conditional-WHILE decode request, i.e. everything
// the body's termination logic depends on. A VALID EOS id is mandatory (see
// qwen_validate_while_request): without one the loop could only stop on the token
// budget, so a chunk would keep executing steps after the model emitted
// end-of-sequence and the host would have to discard them.
struct QwenWhileRequest {
    bool requested{false};
    uint32_t chunk{0};         // tokens one graph launch may execute (>= 1)
    uint32_t eos_token{0};     // model EOS id, or a value >= vocab for "none"
    uint32_t vocab{0};         // model vocabulary
    uint32_t position{0};      // absolute position of the chunk's first token
    uint32_t max_context{0};   // KV/context capacity in tokens
    [[nodiscard]] bool eos_enabled() const noexcept { return eos_token < vocab; }
    [[nodiscard]] uint32_t first_step() const noexcept { return position; }
};

// Device-free validation of the request. Failure means the engine must refuse the
// chunked path (StatusCode::invalid_argument / unsupported) instead of building a
// body whose terminations could not be honoured.
[[nodiscard]] Status qwen_validate_while_request(const QwenWhileRequest &request) noexcept;

// One-line, harness-parsable description of a request/refusal. Kept next to the
// validation so a log line and the engine's reason string can never disagree.
[[nodiscard]] luisa::string qwen_while_request_summary(const QwenWhileRequest &request) noexcept;

// Result of the instantiate/execute probe. `supported` is the ONLY field the
// engine tests; the rest exists so a refusal can be reported with the measured
// numbers instead of a guess.
struct QwenCudaWhileProbe {
    bool extension{false};   // device.extension<CudaGraphExt>() != nullptr
    bool capabilities{false};// CudaGraphExt::Capabilities::conditional_nodes
    bool attempted{false};   // a WHILE graph was actually built and launched
    bool supported{false};   // body executed exactly 3x, replay 3x, zero-trip 0x
    uint32_t iterations{0};  // body executions observed on the first launch (expected 3)
    uint32_t replays{0};     // body executions of the replayed launch (expected 3)
    uint32_t zero_trip_iterations{0};// body executions of the default-0 launch (expected 0)
    uint32_t body_nodes{0};  // kernel nodes the probe's body graph accepted
    double instantiate_ms{0.0};
    double execute_ms{0.0};
    luisa::string detail;    // refusal/failure text (empty on success)
    [[nodiscard]] luisa::string summary() const noexcept;
};

// Build, instantiate, execute (twice) and destroy a two-graph WHILE probe: one
// do-while graph whose body counts down 5 -> 2 with the launch default assigned
// (3 executions) and one zero-trip graph whose default is 0. Never throws; any
// failure is a normal "not supported" answer with `detail` set.
[[nodiscard]] QwenCudaWhileProbe qwen_probe_cuda_graph_while(
    compute::Device &device, compute::Stream &stream) noexcept;

}// namespace luisa::qwen
