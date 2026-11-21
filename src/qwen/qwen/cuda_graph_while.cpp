// Conditional CUDA graph WHILE decode chunk -- request validation + probe.
// See qwen/cuda_graph_while.h for the contract. Nothing here is on any hot path:
// validation is device-free and the probe runs ONLY when the caller explicitly
// asked for the chunked graph (--cuda-graph-while / --graph-flight), so a default
// run pays nothing for it.

#include "qwen/cuda_graph_while.h"

#include <cstring>

#include <luisa/core/clock.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/buffer.h>

#include <luisa/backends/ext/cuda/cuda_graph_ext.h>

namespace luisa::qwen {

[[nodiscard]] Status qwen_validate_while_request(const QwenWhileRequest &request) noexcept {
    if (!request.requested) { return Status::success(); }
    if (request.vocab == 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "conditional WHILE decode requires a non-zero vocabulary");
    }
    if (request.max_context == 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "conditional WHILE decode requires a non-zero context capacity");
    }
    if (request.chunk == 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "conditional WHILE decode requires a chunk of at least 1 token");
    }
    if (request.position >= request.max_context) {
        return Status::error(StatusCode::invalid_argument,
                             "conditional WHILE decode cannot start at or past max_context");
    }
    if (request.chunk > request.max_context - request.position) {
        return Status::error(StatusCode::invalid_argument,
                             "conditional WHILE decode chunk exceeds the remaining context",
                             "position " + luisa::string{std::to_string(request.position)} +
                                 ", chunk " + luisa::string{std::to_string(request.chunk)} +
                                 ", max_context " + luisa::string{std::to_string(request.max_context)});
    }
    // An EOS id is MANDATORY for the chunked path: without it the loop would only
    // stop on the token budget, i.e. it would keep executing steps after the model
    // emitted end-of-sequence and the host would have to discard them. Refusing the
    // request is the honest alternative (the per-token path stays available).
    if (request.eos_token >= request.vocab) {
        return Status::error(StatusCode::unsupported,
                             "conditional WHILE decode requires a valid EOS token id "
                             "(QwenEngineOptions::dsl_eos_token_id)",
                             "eos " + luisa::string{std::to_string(request.eos_token)} +
                                 ", vocab " + luisa::string{std::to_string(request.vocab)});
    }
    return Status::success();
}

[[nodiscard]] luisa::string qwen_while_request_summary(const QwenWhileRequest &request) noexcept {
    return luisa::format("requested={} chunk={} eos={} vocab={} position={} max_context={}",
                         request.requested ? 1 : 0, request.chunk, request.eos_token,
                         request.vocab, request.position, request.max_context);
}

[[nodiscard]] luisa::string QwenCudaWhileProbe::summary() const noexcept {
    if (!attempted) {
        return luisa::format("extension={} conditional-nodes={} supported=0 reason='{}'",
                             extension ? 1 : 0, capabilities ? 1 : 0, detail);
    }
    return luisa::format("extension={} conditional-nodes={} attempted=1 body-nodes={} "
                         "iterations={} replay={} zero-trip={} instantiate-ms={} execute-ms={} "
                         "supported={} reason='{}'",
                         extension ? 1 : 0, capabilities ? 1 : 0, body_nodes,
                         iterations, replays, zero_trip_iterations,
                         instantiate_ms, execute_ms, supported ? 1 : 0, detail);
}

namespace {

// One probe graph: parent = [two uploads] -> WHILE(default, assign_default) whose
// body runs `countdown` once per iteration. Returns the observed iteration count
// (0 on any failure; `detail` then explains which CUDA call refused).
struct WhileProbeGraph {
    uint32_t first_iterations{0};
    uint32_t iterations{0};
    uint32_t body_nodes{0};
    double instantiate_ms{0.0};
    bool ok{false};
    luisa::string detail;
};

[[nodiscard]] WhileProbeGraph run_probe_graph(
    compute::Device &device, compute::Stream &stream, compute::CudaGraphExt *ext,
    uint32_t default_value, bool expect_body) noexcept {
    using namespace luisa::compute;
    WhileProbeGraph result;
    // Body: counter -= 1, iterations += 1, re-arm the loop while counter > target.
    // `handle` arrives as a kernel ARGUMENT: the body is built after the handle
    // exists, which is exactly the production pattern (the body is only ever
    // populated through build_body_graph()).
    Kernel1D countdown = [](BufferUInt counter, BufferUInt iterations,
                            Var<uint64_t> handle, Var<uint> target) noexcept {
        set_block_size(32u, 1u, 1u);
        $if (thread_id().x == 0u) {
            auto next = counter.read(0u) - 1u;
            counter.write(0u, next);
            iterations.write(0u, iterations.read(0u) + 1u);
            set_graph_conditional(handle, (next > target).cast<uint>());
        };
    };
    // Zero-trip body: must never run when the launch default is 0.
    Kernel1D bump = [](BufferUInt iterations) noexcept {
        set_block_size(32u, 1u, 1u);
        $if (thread_id().x == 0u) {
            iterations.write(0u, iterations.read(0u) + 1u);
        };
    };
    auto countdown_shader = device.compile(countdown);
    auto bump_shader = device.compile(bump);
    if (!countdown_shader || !countdown_shader.compile_ok() ||
        !bump_shader || !bump_shader.compile_ok()) {
        result.detail = "probe shader compile failed";
        return result;
    }
    auto counter = device.create_buffer<uint32_t>(1u);
    auto iterations = device.create_buffer<uint32_t>(1u);
    if (!counter || !iterations) {
        result.detail = "probe buffer allocation failed";
        return result;
    }
    // Upload staging must outlive the launch: a graph upload node reads its host
    // pointer when it EXECUTES, not when it is recorded.
    luisa::vector<uint32_t> initial_counter{5u};
    luisa::vector<uint32_t> initial_iterations{0u};
    auto cmdlist = CommandList::create();
    cmdlist << counter.view().copy_from(luisa::span{initial_counter});
    cmdlist << iterations.view().copy_from(luisa::span{initial_iterations});
    auto graph = ext->create_graph(cmdlist.commit().command_list());
    if (graph.handle().handle == compute::CudaGraphExt::invalid_handle) {
        result.detail = "create_graph refused the probe command list";
        return result;
    }
    auto graph_handle = graph.handle().handle;
    auto handle = ext->create_conditional_handle(graph_handle, default_value, true);
    if (handle == 0u) {
        result.detail = "create_conditional_handle refused";
        return result;
    }
    luisa::vector<compute::CudaGraphExt::GraphHandle> bodies(1u, compute::CudaGraphExt::invalid_handle);
    size_t conditional_index = ~size_t{0u};
    if (!ext->add_conditional_node(graph_handle, compute::CudaGraphExt::ConditionalType::WHILE,
                                   handle, 1u, luisa::span{bodies}, &conditional_index)) {
        result.detail = "add_conditional_node(WHILE) refused";
        return result;
    }
    if (bodies[0] == compute::CudaGraphExt::invalid_handle) {
        result.detail = "WHILE node did not hand back a body graph";
        return result;
    }
    for (size_t i = 0u; i < conditional_index; i++) {
        if (!ext->add_dependency(graph_handle, i, conditional_index)) {
            result.detail = "add_dependency refused";
            return result;
        }
    }
    auto body_cmdlist = CommandList::create();
    if (default_value != 0u) {
        body_cmdlist << countdown_shader(counter.view(), iterations.view(), handle, 2u).dispatch(1u);
    } else {
        body_cmdlist << bump_shader(iterations.view()).dispatch(1u);
    }
    if (!ext->build_body_graph(bodies[0], body_cmdlist.commit().command_list())) {
        result.detail = "build_body_graph refused the probe body";
        return result;
    }
    auto instantiate_clock = Clock{};
    auto exec = ext->instantiate(graph_handle);
    result.instantiate_ms = instantiate_clock.toc();
    if (exec.handle().handle == compute::CudaGraphExt::invalid_handle) {
        result.detail = "instantiate refused";
        return result;
    }
    result.body_nodes = 1u;
    // Two launches: the second is the REPLAY check (the parent's uploads run again,
    // so the trip count must be identical, which is what proves the assigned default
    // is re-applied per launch instead of being consumed by the first one).
    for (auto replay = 0; replay < 2; replay++) {
        ext->launch(exec.handle().handle, stream.handle());
        stream << compute::synchronize();
        luisa::vector<uint32_t> observed_iterations(1u, 0u);
        luisa::vector<uint32_t> observed_counter(1u, 0u);
        stream << iterations.view().copy_to(luisa::span{observed_iterations})
               << counter.view().copy_to(luisa::span{observed_counter})
               << compute::synchronize();
        if (replay == 0) { result.first_iterations = observed_iterations[0]; }
        result.iterations = observed_iterations[0];
        if (default_value != 0u) {
            // counter 5 -> 2 is exactly 3 body executions, and the parent upload
            // re-runs on every launch so both iterations agree.
            if (result.iterations != 3u || observed_counter[0] != 2u) {
                result.detail = luisa::format("unexpected trip count: iterations {} (want 3), "
                                              "counter {} (want 2)", result.iterations,
                                              observed_counter[0]);
                return result;
            }
        } else if (result.iterations != 0u) {
            result.detail = luisa::format("a WHILE with launch default 0 executed its body {} time(s)",
                                          result.iterations);
            return result;
        }
        (void)expect_body;
    }
    result.ok = true;
    return result;
}

}// namespace

[[nodiscard]] QwenCudaWhileProbe qwen_probe_cuda_graph_while(
    compute::Device &device, compute::Stream &stream) noexcept {
    QwenCudaWhileProbe probe;
    auto *ext = device.extension<compute::CudaGraphExt>();
    if (ext == nullptr) {
        probe.detail = "CudaGraphExt is not available on this device";
        return probe;
    }
    probe.extension = true;
    const auto caps = ext->capabilities();
    probe.capabilities = caps.conditional_nodes;
    if (!caps.conditional_nodes) {
        probe.detail = luisa::format("CudaGraphExt reports no conditional-node support "
                                     "(toolkit CUDA_VERSION {})", caps.cuda_version);
        return probe;
    }
    probe.attempted = true;
    // 1) do-while semantics: the assigned launch default 1 runs the body until the
    //    body itself clears the condition (the production termination shape).
    auto graph_clock = Clock{};
    auto do_while = run_probe_graph(device, stream, ext, 1u, true);
    if (!do_while.ok) {
        probe.detail = "do-while: " + do_while.detail;
        return probe;
    }
    probe.iterations = do_while.first_iterations;
    probe.replays = do_while.iterations;
    probe.body_nodes = do_while.body_nodes;
    probe.instantiate_ms = do_while.instantiate_ms;
    // 2) zero-trip semantics: a launch default of 0 must skip the body entirely.
    //    This is what makes an already-finished chunk cheap instead of one token
    //    long (and it proves the default is really re-assigned per launch).
    auto zero_trip = run_probe_graph(device, stream, ext, 0u, false);
    if (!zero_trip.ok) {
        probe.detail = "zero-trip: " + zero_trip.detail;
        return probe;
    }
    probe.zero_trip_iterations = zero_trip.iterations;
    probe.execute_ms = graph_clock.toc();
    probe.supported = true;
    return probe;
}

}// namespace luisa::qwen
